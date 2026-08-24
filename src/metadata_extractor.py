"""
Extract metadata from document chunks for better filtering.

Extracts section numbers, schedules, and other structural information
to improve retrieval quality through metadata filtering.
"""

import re
from typing import Dict, List, Set


class MetadataExtractor:
    """Extract metadata from document chunks."""

    # Tax section patterns
    SECTION_PATTERN = r'\b(?:section|sec|s\.|u/s)\s*\.?([0-9]{1,3}[A-Z]?(?:\([A-Z0-9]+\))?)'
    SCHEDULE_PATTERN = r'\b(?:schedule|sch|sch\.)\s*([A-Z]{1,2})'
    FORM_PATTERN = r'\b(?:ITR|form)\s*[-]?([0-9]{1,2})'
    RELIEF_PATTERN = r'\b(?:relief)\s*(?:u/s|under section)\s*([0-9]{1,3})'

    @staticmethod
    def extract_sections(text: str) -> Set[str]:
        """
        Extract section numbers from text.

        Args:
            text: Text to search

        Returns:
            Set of section numbers (e.g., {'80C', '80D', '89'})
        """
        sections = set()

        # Find all section references
        for match in re.finditer(MetadataExtractor.SECTION_PATTERN, text, re.IGNORECASE):
            section = match.group(1).upper()
            sections.add(section)

        return sections

    @staticmethod
    def extract_schedules(text: str) -> Set[str]:
        """
        Extract schedule references from text.

        Args:
            text: Text to search

        Returns:
            Set of schedules (e.g., {'AA', 'AB', 'TDS'})
        """
        schedules = set()

        for match in re.finditer(MetadataExtractor.SCHEDULE_PATTERN, text, re.IGNORECASE):
            schedule = match.group(1).upper()
            schedules.add(schedule)

        return schedules

    @staticmethod
    def extract_forms(text: str) -> Set[str]:
        """
        Extract ITR form numbers from text.

        Args:
            text: Text to search

        Returns:
            Set of forms (e.g., {'1', '2', '3'})
        """
        forms = set()

        for match in re.finditer(MetadataExtractor.FORM_PATTERN, text, re.IGNORECASE):
            form = match.group(1)
            forms.add(form)

        return forms

    @staticmethod
    def extract_metadata(text: str) -> Dict[str, any]:
        """
        Extract all metadata from chunk text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with extracted metadata
        """
        sections = MetadataExtractor.extract_sections(text)
        schedules = MetadataExtractor.extract_schedules(text)
        forms = MetadataExtractor.extract_forms(text)

        return {
            "sections": list(sections),
            "schedules": list(schedules),
            "forms": list(forms),
            "has_deduction": any(s.startswith('80') for s in sections),
            "has_relief": any(s.startswith('89') for s in sections),
            "has_income": any(s in sections for s in ['115AB', '115AC', '115AD']),
        }

    @staticmethod
    def get_section_category(section: str) -> str:
        """
        Get category for a section number.

        Args:
            section: Section number (e.g., '80C')

        Returns:
            Category (e.g., 'deduction', 'relief', 'income')
        """
        section = section.upper().strip()

        if section.startswith('80'):
            return 'deduction'
        elif section.startswith('89'):
            return 'relief'
        elif section.startswith('115'):
            return 'income'
        else:
            return 'other'
