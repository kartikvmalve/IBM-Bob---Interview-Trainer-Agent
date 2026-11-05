"""
IBM Interview Trainer Agent — Resume NLP Parser
Extracts skills, experience, education, projects, and generates
a structured candidate profile from raw resume text.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.granite_client import get_granite_client

logger = logging.getLogger(__name__)

# ── Skill taxonomy (fast keyword matching, no model needed) ───────────────────
TECH_SKILLS = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "scala", "r", "matlab", "sql", "bash", "php", "ruby",
    # Frameworks / Libraries
    "react", "angular", "vue", "django", "flask", "fastapi", "spring", "nodejs",
    "express", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
    "numpy", "spark", "hadoop", "kafka",
    # Cloud & DevOps
    "aws", "azure", "gcp", "ibm cloud", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "github actions", "ci/cd", "linux", "git",
    "openshift", "ibm watsonx", "watson",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
    "dynamodb", "sqlite", "oracle", "db2",
    # Concepts
    "machine learning", "deep learning", "nlp", "computer vision", "rag",
    "microservices", "rest api", "graphql", "agile", "scrum", "devops",
    "system design", "data structures", "algorithms",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving", "critical thinking",
    "time management", "adaptability", "creativity", "collaboration", "mentoring",
    "project management", "stakeholder management", "presentation",
}


@dataclass
class CandidateProfile:
    raw_text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    total_experience_years: float = 0.0
    current_role: str = ""
    tech_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    ai_summary: str = ""         # LLM-generated summary
    suggested_role: str = ""
    suggested_level: str = ""

    def to_summary(self) -> str:
        """Compact string used as resume context in prompts."""
        parts = []
        if self.name:
            parts.append(f"Name: {self.name}")
        if self.current_role:
            parts.append(f"Current Role: {self.current_role}")
        if self.total_experience_years:
            parts.append(f"Experience: {self.total_experience_years} years")
        if self.tech_skills:
            parts.append(f"Tech Skills: {', '.join(self.tech_skills[:15])}")
        if self.soft_skills:
            parts.append(f"Soft Skills: {', '.join(self.soft_skills[:5])}")
        if self.education:
            parts.append(f"Education: {'; '.join(self.education[:2])}")
        if self.companies:
            parts.append(f"Companies: {', '.join(self.companies[:3])}")
        if self.certifications:
            parts.append(f"Certifications: {', '.join(self.certifications[:3])}")
        return "\n".join(parts)


class ResumeParser:
    """
    Two-pass resume parser:
    Pass 1 — fast regex/keyword extraction (no LLM call, always runs)
    Pass 2 — IBM Granite LLM for a rich narrative summary (optional, async)
    """

    def __init__(self) -> None:
        self._llm = get_granite_client()

    def parse_sync(self, text: str) -> CandidateProfile:
        """Synchronous pass-1 extraction — safe to call from non-async context."""
        profile = CandidateProfile(raw_text=text)
        profile.name = self._extract_name(text)
        profile.email = self._extract_email(text)
        profile.phone = self._extract_phone(text)
        profile.tech_skills = self._extract_skills(text, TECH_SKILLS)
        profile.soft_skills = self._extract_skills(text, SOFT_SKILLS)
        profile.total_experience_years = self._estimate_experience(text)
        profile.education = self._extract_education(text)
        profile.companies = self._extract_companies(text)
        profile.certifications = self._extract_certifications(text)
        profile.projects = self._extract_projects(text)
        profile.current_role = self._extract_current_role(text)
        profile.suggested_level = _suggest_level(profile.total_experience_years)
        profile.suggested_role = self._suggest_role(profile)
        return profile

    async def parse(self, text: str) -> CandidateProfile:
        """Full async parse: regex extraction + LLM narrative summary."""
        profile = self.parse_sync(text)
        profile.ai_summary = await self._generate_ai_summary(profile)
        return profile

    # ── Pass-1 extractors ─────────────────────────────────────────────────

    @staticmethod
    def _extract_email(text: str) -> str:
        m = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
        return m.group(0) if m else ""

    @staticmethod
    def _extract_phone(text: str) -> str:
        m = re.search(r"(\+?[\d\s\-().]{10,15})", text)
        return m.group(0).strip() if m else ""

    @staticmethod
    def _extract_name(text: str) -> str:
        """Heuristic: first non-empty line that looks like a name (<=4 words, no @)."""
        for line in text.splitlines():
            line = line.strip()
            if not line or "@" in line or len(line) > 60:
                continue
            words = line.split()
            if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                return line
        return ""

    @staticmethod
    def _extract_skills(text: str, skill_set: set[str]) -> list[str]:
        text_lower = text.lower()
        found = []
        for skill in sorted(skill_set):
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                found.append(skill)
        return found

    @staticmethod
    def _estimate_experience(text: str) -> float:
        """Look for explicit 'X years' mentions or count date ranges."""
        # Explicit mention
        m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:experience|exp)", text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        # Count year ranges like 2018-2022 or 2018–2023
        ranges = re.findall(r"\b(20\d{2})\s*[-–]\s*(20\d{2}|present|current)\b", text, re.IGNORECASE)
        total = 0.0
        for start, end in ranges:
            try:
                end_year = 2026 if end.lower() in ("present", "current") else int(end)
                total += max(0, end_year - int(start))
            except ValueError:
                pass
        return round(min(total, 40), 1)

    @staticmethod
    def _extract_education(text: str) -> list[str]:
        degrees = re.findall(
            r"(?:B\.?Tech|B\.?E|B\.?Sc|M\.?Tech|M\.?Sc|M\.?S|MBA|Ph\.?D|Bachelor|Master|Doctor)[^\n]{0,80}",
            text, re.IGNORECASE,
        )
        return [d.strip() for d in degrees[:3]]

    @staticmethod
    def _extract_companies(text: str) -> list[str]:
        """Extract lines that look like company names (near @ or in experience sections)."""
        lines = text.splitlines()
        companies = []
        keywords = ("inc", "ltd", "llc", "pvt", "corp", "technologies", "solutions",
                    "systems", "services", "consulting", "software", "labs", "ibm",
                    "google", "microsoft", "amazon", "infosys", "tcs", "wipro", "hcl")
        for line in lines:
            l = line.strip()
            if any(k in l.lower() for k in keywords) and len(l) < 80:
                companies.append(l)
        return companies[:5]

    @staticmethod
    def _extract_certifications(text: str) -> list[str]:
        certs = re.findall(
            r"(?:AWS|Azure|GCP|IBM|Google|Oracle|Cisco|PMP|CKA|CKS|Terraform|Kubernetes)"
            r"[^\n]{0,60}(?:Certified|Certificate|Certification|Associate|Professional|Expert)",
            text, re.IGNORECASE,
        )
        certs += re.findall(
            r"(?:Certified|Certificate)[^\n]{0,60}(?:Engineer|Developer|Architect|Professional|Associate)",
            text, re.IGNORECASE,
        )
        return list(dict.fromkeys(c.strip() for c in certs))[:5]

    @staticmethod
    def _extract_projects(text: str) -> list[str]:
        projects = re.findall(r"(?:Project|Built|Developed|Created|Implemented)[:\s]+([^\n]{10,100})", text, re.IGNORECASE)
        return [p.strip() for p in projects[:5]]

    @staticmethod
    def _extract_current_role(text: str) -> str:
        roles = re.findall(
            r"(?:Software|Data|ML|DevOps|Cloud|Product|Backend|Frontend|Full.?Stack|"
            r"QA|Test|SRE|Platform|Principal|Senior|Lead|Junior|Associate)\s+"
            r"(?:Engineer|Developer|Scientist|Architect|Manager|Analyst|Intern)[^\n]{0,30}",
            text, re.IGNORECASE,
        )
        return roles[0].strip() if roles else ""

    @staticmethod
    def _suggest_role(profile: "CandidateProfile") -> str:
        skills_lower = {s.lower() for s in profile.tech_skills}
        if any(s in skills_lower for s in ("tensorflow", "pytorch", "machine learning", "deep learning", "nlp")):
            return "Data Scientist / ML Engineer"
        if any(s in skills_lower for s in ("kubernetes", "docker", "terraform", "ansible", "ci/cd")):
            return "DevOps / Cloud Engineer"
        if any(s in skills_lower for s in ("react", "angular", "vue")):
            return "Frontend Developer"
        if any(s in skills_lower for s in ("django", "flask", "fastapi", "spring")):
            return "Backend Developer"
        if profile.current_role:
            return profile.current_role
        return "Software Engineer"

    # ── Pass-2: LLM narrative ─────────────────────────────────────────────

    async def _generate_ai_summary(self, profile: CandidateProfile) -> str:
        prompt = f"""<|system|>
You are a professional resume analyst. Write a concise, insightful candidate summary.
<|user|>
Analyse this resume data and write a 3-4 sentence professional summary:

Name: {profile.name or "Candidate"}
Experience: {profile.total_experience_years} years
Current Role: {profile.current_role or "Not specified"}
Tech Skills: {', '.join(profile.tech_skills[:12]) or "Not detected"}
Education: {'; '.join(profile.education[:2]) or "Not specified"}
Companies: {', '.join(profile.companies[:3]) or "Not specified"}
Certifications: {', '.join(profile.certifications[:3]) or "None"}

Write a professional summary highlighting key strengths and interview-relevant qualities.
<|assistant|>
"""
        try:
            return await self._llm.generate(prompt, max_new_tokens=200)
        except Exception as exc:
            logger.warning("LLM summary failed, using fallback: %s", exc)
            return profile.to_summary()


def _suggest_level(years: float) -> str:
    if years < 1:
        return "fresher"
    if years < 3:
        return "junior"
    if years < 6:
        return "mid"
    if years < 10:
        return "senior"
    return "lead"


# ── Singleton ─────────────────────────────────────────────────────────────────
_parser: ResumeParser | None = None


def get_resume_parser() -> ResumeParser:
    global _parser
    if _parser is None:
        _parser = ResumeParser()
    return _parser
