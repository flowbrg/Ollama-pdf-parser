import fitz
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class TextStyle(Enum):
    NORMAL = "normal"
    BOLD = "bold"
    ITALIC = "italic"
    HEADER = "header"
    SUBSCRIPT = "subscript"
    SUPERSCRIPT = "superscript"

@dataclass
class TextElement:
    """Represents a text element with formatting"""
    text: str
    bbox: Tuple[float, float, float, float]
    font: str
    size: float
    style: TextStyle
    color: int
    is_formula: bool = False

@dataclass
class ExtractedContent:
    """Container for extracted text content"""
    elements: List[TextElement]
    structure: Dict
    formulas: List[str]
    raw_text: str

class TextExtractor:
    """Extract text directly from PDF when possible"""
    
    def __init__(self):
        self.formula_patterns = self._compile_formula_patterns()
        self.header_size_threshold = 1.2  # Relative to average font size
        
    def extract_structured_text(self, page: fitz.Page) -> ExtractedContent:
        """Extract text with position, font, style info"""
        text_dict = page.get_text("dict")
        elements = []
        formulas = []
        
        # Calculate average font size for header detection
        avg_font_size = self._calculate_average_font_size(text_dict)
        
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    
                    # Determine style
                    style = self._determine_text_style(span, avg_font_size)
                    
                    # Check if text contains formulas
                    is_formula = self._contains_formula(text)
                    if is_formula:
                        formulas.extend(self._extract_formula_parts(text))
                    
                    element = TextElement(
                        text=text,
                        bbox=span["bbox"],
                        font=span["font"],
                        size=span["size"],
                        style=style,
                        color=span.get("color", 0),
                        is_formula=is_formula
                    )
                    elements.append(element)
        
        # Extract document structure
        structure = self._analyze_document_structure(elements)
        
        # Get raw text for fallback
        raw_text = page.get_text()
        
        return ExtractedContent(
            elements=elements,
            structure=structure,
            formulas=formulas,
            raw_text=raw_text
        )
    
    def extract_mathematical_formulas(self, page: fitz.Page) -> List[str]:
        """Extract LaTeX/MathML formulas if embedded"""
        formulas = []
        
        # Try to extract embedded LaTeX
        text = page.get_text()
        latex_formulas = self._extract_latex_formulas(text)
        formulas.extend(latex_formulas)
        
        # Look for mathematical symbols and patterns
        math_formulas = self._extract_math_patterns(text)
        formulas.extend(math_formulas)
        
        return list(set(formulas))  # Remove duplicates
    
    def preserve_formatting(self, elements: List[TextElement]) -> str:
        """Maintain headers, lists, emphasis from PDF structure"""
        markdown_lines = []
        current_list_level = 0
        
        # Group elements by lines based on y-coordinates
        lines = self._group_elements_by_line(elements)
        
        for line_elements in lines:
            line_text = self._format_line_elements(line_elements)
            
            if not line_text.strip():
                markdown_lines.append("")
                continue
            
            # Detect list items
            if self._is_list_item(line_text):
                list_marker = self._get_list_marker(line_text)
                clean_text = self._clean_list_text(line_text)
                markdown_lines.append(f"{list_marker} {clean_text}")
            
            # Detect headers
            elif any(elem.style == TextStyle.HEADER for elem in line_elements):
                header_level = self._determine_header_level(line_elements)
                markdown_lines.append(f"{'#' * header_level} {line_text}")
            
            # Regular text with inline formatting
            else:
                formatted_text = self._apply_inline_formatting(line_elements)
                markdown_lines.append(formatted_text)
        
        return "\n".join(markdown_lines)
    
    def _compile_formula_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for formula detection"""
        patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\\\[[^\]]+\\\]',  # LaTeX display math
            r'\\begin\{equation\}.*?\\end\{equation\}',  # LaTeX equations
            r'[a-zA-Z][_^][0-9a-zA-Z]+',  # Subscripts/superscripts
            r'[∫∑∏√±×÷≤≥≠≈∞∂∇]',  # Mathematical symbols
            r'\b(sin|cos|tan|log|ln|exp|lim|int|sum|prod)\s*\(',  # Math functions
            r'[a-zA-Z]\s*[=]\s*[0-9a-zA-Z\s\+\-\*/\^]+',  # Simple equations
        ]
        return [re.compile(p, re.DOTALL | re.IGNORECASE) for p in patterns]
    
    def _calculate_average_font_size(self, text_dict: Dict) -> float:
        """Calculate average font size for header detection"""
        sizes = []
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        sizes.append(span["size"])
        return sum(sizes) / len(sizes) if sizes else 12.0
    
    def _determine_text_style(self, span: Dict, avg_font_size: float) -> TextStyle:
        """Determine text style from span properties"""
        flags = span["flags"]
        size = span["size"]
        
        # Check for header (larger than average)
        if size > avg_font_size * self.header_size_threshold:
            return TextStyle.HEADER
        
        # Check font flags (bold, italic)
        if flags & 2**4:  # Bold flag
            return TextStyle.BOLD
        if flags & 2**1:  # Italic flag
            return TextStyle.ITALIC
        if flags & 2**0:  # Superscript
            return TextStyle.SUPERSCRIPT
        if flags & 2**2:  # Subscript (not standard, custom detection)
            return TextStyle.SUBSCRIPT
        
        return TextStyle.NORMAL
    
    def _contains_formula(self, text: str) -> bool:
        """Check if text contains mathematical formulas"""
        return any(pattern.search(text) for pattern in self.formula_patterns)
    
    def _extract_formula_parts(self, text: str) -> List[str]:
        """Extract formula components from text"""
        formulas = []
        for pattern in self.formula_patterns:
            matches = pattern.findall(text)
            formulas.extend(matches)
        return formulas
    
    def _extract_latex_formulas(self, text: str) -> List[str]:
        """Extract LaTeX formulas from text"""
        patterns = [
            r'\$([^$]+)\$',  # Inline math
            r'\\\[([^\]]+)\\\]',  # Display math
            r'\\begin\{equation\}(.*?)\\end\{equation\}',  # Equations
        ]
        
        formulas = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            formulas.extend(matches)
        
        return formulas
    
    def _extract_math_patterns(self, text: str) -> List[str]:
        """Extract mathematical patterns from text"""
        # Look for equation-like patterns
        equation_pattern = r'[a-zA-Z]\s*[=]\s*[^.!?]*'
        equations = re.findall(equation_pattern, text)
        
        # Filter for likely mathematical content
        math_equations = []
        for eq in equations:
            if re.search(r'[0-9+\-*/^√∫∑]', eq):
                math_equations.append(eq.strip())
        
        return math_equations
    
    def _analyze_document_structure(self, elements: List[TextElement]) -> Dict:
        """Analyze document structure (headers, paragraphs, etc.)"""
        structure = {
            "headers": [],
            "paragraphs": [],
            "lists": [],
            "formulas": []
        }
        
        for elem in elements:
            if elem.style == TextStyle.HEADER:
                structure["headers"].append(elem.text)
            elif elem.is_formula:
                structure["formulas"].append(elem.text)
        
        return structure
    
    def _group_elements_by_line(self, elements: List[TextElement]) -> List[List[TextElement]]:
        """Group text elements by line based on y-coordinates"""
        # Sort by y-coordinate
        sorted_elements = sorted(elements, key=lambda e: e.bbox[1])
        
        lines = []
        current_line = []
        current_y = None
        y_tolerance = 5  # Pixels
        
        for elem in sorted_elements:
            elem_y = elem.bbox[1]
            
            if current_y is None or abs(elem_y - current_y) <= y_tolerance:
                current_line.append(elem)
                current_y = elem_y
            else:
                if current_line:
                    # Sort by x-coordinate within line
                    current_line.sort(key=lambda e: e.bbox[0])
                    lines.append(current_line)
                current_line = [elem]
                current_y = elem_y
        
        if current_line:
            current_line.sort(key=lambda e: e.bbox[0])
            lines.append(current_line)
        
        return lines
    
    def _format_line_elements(self, elements: List[TextElement]) -> str:
        """Format elements within a line"""
        return " ".join(elem.text for elem in elements)
    
    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item"""
        list_patterns = [
            r'^\s*[•·◦▪▫]\s+',  # Bullet points
            r'^\s*[0-9]+\.\s+',  # Numbered lists
            r'^\s*[a-zA-Z]\.\s+',  # Lettered lists
            r'^\s*[-*+]\s+',  # Dash/asterisk bullets
        ]
        return any(re.match(pattern, text) for pattern in list_patterns)
    
    def _get_list_marker(self, text: str) -> str:
        """Get appropriate markdown list marker"""
        if re.match(r'^\s*[0-9]+\.', text):
            return "1."
        else:
            return "-"
    
    def _clean_list_text(self, text: str) -> str:
        """Remove list markers from text"""
        return re.sub(r'^\s*([•·◦▪▫]|[0-9]+\.|[a-zA-Z]\.|[-*+])\s+', '', text)
    
    def _determine_header_level(self, elements: List[TextElement]) -> int:
        """Determine header level (1-6) based on font size"""
        max_size = max(elem.size for elem in elements)
        
        # Simple mapping based on font size
        if max_size >= 24:
            return 1
        elif max_size >= 20:
            return 2
        elif max_size >= 16:
            return 3
        elif max_size >= 14:
            return 4
        elif max_size >= 12:
            return 5
        else:
            return 6
    
    def _apply_inline_formatting(self, elements: List[TextElement]) -> str:
        """Apply markdown formatting to inline elements"""
        formatted_parts = []
        
        for elem in elements:
            text = elem.text
            
            if elem.style == TextStyle.BOLD:
                text = f"**{text}**"
            elif elem.style == TextStyle.ITALIC:
                text = f"*{text}*"
            elif elem.is_formula:
                text = f"`{text}`"  # Use code formatting for inline math
            
            formatted_parts.append(text)
        
        return " ".join(formatted_parts)