"""Markdown parser with chapter detection."""

import re
from dataclasses import dataclass
from typing import List, Optional

from ..formats.base import Segment


@dataclass
class MarkdownSection:
    """A section parsed from markdown."""

    level: int  # Header level (1-6)
    title: str
    content: str


class MarkdownParser:
    """Parser for Markdown input with header detection."""

    def __init__(
        self,
        header_pause_ms: int = 1000,
        paragraph_pause_ms: int = 500,
        include_headers: bool = True,
    ):
        """
        Initialize parser.

        Args:
            header_pause_ms: Pause after headers
            paragraph_pause_ms: Pause between paragraphs
            include_headers: Whether to include headers in output
        """
        self.header_pause_ms = header_pause_ms
        self.paragraph_pause_ms = paragraph_pause_ms
        self.include_headers = include_headers

    def parse_sections(self, text: str) -> List[MarkdownSection]:
        """
        Parse markdown into sections by headers.

        Args:
            text: Raw markdown content

        Returns:
            List of MarkdownSection objects
        """
        sections = []
        lines = text.split('\n')

        current_level = 0
        current_title = ""
        current_content = []

        for line in lines:
            # Check for header
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                # Save previous section
                if current_title or current_content:
                    sections.append(
                        MarkdownSection(
                            level=current_level,
                            title=current_title,
                            content='\n'.join(current_content).strip(),
                        )
                    )

                current_level = len(match.group(1))
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # Don't forget last section
        if current_title or current_content:
            sections.append(
                MarkdownSection(
                    level=current_level,
                    title=current_title,
                    content='\n'.join(current_content).strip(),
                )
            )

        return sections

    def parse(self, text: str) -> List[Segment]:
        """
        Parse markdown into segments.

        Args:
            text: Raw markdown content

        Returns:
            List of Segment objects
        """
        segments = []

        # Clean markdown formatting
        clean_text = self._clean_markdown(text)

        sections = self.parse_sections(clean_text)

        for section in sections:
            # Add header as segment
            if section.title and self.include_headers:
                segments.append(
                    Segment(
                        text=section.title,
                        pause_after_ms=self.header_pause_ms,
                        metadata={"type": "header", "level": section.level},
                    )
                )

            # Add content paragraphs
            if section.content:
                paragraphs = re.split(r'\n\n+', section.content)
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        segments.append(
                            Segment(
                                text=para,
                                pause_after_ms=self.paragraph_pause_ms,
                            )
                        )

        return segments

    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting but preserve structure."""
        # Remove bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)

        # Remove links but keep text
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)

        # Remove inline code
        text = re.sub(r'`(.+?)`', r'\1', text)

        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)

        # Remove images
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

        # Remove horizontal rules
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)

        # Remove blockquotes
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

        # Clean up list markers
        text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)

        return text

    def parse_file(self, path: str, encoding: str = "utf-8") -> List[Segment]:
        """
        Parse a markdown file.

        Args:
            path: Path to markdown file
            encoding: File encoding

        Returns:
            List of Segment objects
        """
        with open(path, "r", encoding=encoding) as f:
            text = f.read()
        return self.parse(text)
