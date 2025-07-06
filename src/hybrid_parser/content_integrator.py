from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from difflib import SequenceMatcher

class ContentSource(Enum):
    TEXT_EXTRACTION = "text"
    VISION_MODEL = "vision"
    HYBRID = "hybrid"

@dataclass
class ContentElement:
    """Single piece of content with metadata"""
    content: str
    source: ContentSource
    content_type: str  # "text", "table", "formula", "image", "diagram"
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    order: int = 0

@dataclass
class IntegratedContent:
    """Final integrated content for a page"""
    elements: List[ContentElement]
    markdown: str
    metadata: Dict

class ContentIntegrator:
    """Combine extracted text and vision model outputs"""
    
    def __init__(self):
        self.similarity_threshold = 0.8  # For detecting overlapping content
        self.confidence_bias = {
            ContentSource.TEXT_EXTRACTION: 1.1,  # Prefer text extraction
            ContentSource.VISION_MODEL: 1.0,
            ContentSource.HYBRID: 1.0
        }
    
    def merge_content_streams(self, text_content: Any, vision_content: List[Any], 
                            strategy: str) -> IntegratedContent:
        """Intelligently merge different extraction methods"""
        
        elements = []
        
        # Convert inputs to standardized ContentElement format
        text_elements = self._convert_text_content(text_content)
        vision_elements = self._convert_vision_content(vision_content)
        
        if strategy == "TEXT_ONLY":
            elements = text_elements
        elif strategy == "VISION_ONLY":
            elements = vision_elements
        else:  # HYBRID
            elements = self._merge_hybrid_content(text_elements, vision_elements)
        
        # Resolve conflicts and overlaps
        resolved_elements = self.resolve_conflicts(elements)
        
        # Maintain document flow
        ordered_elements = self.maintain_document_flow(resolved_elements)
        
        # Generate markdown
        markdown = self._elements_to_markdown(ordered_elements)
        
        return IntegratedContent(
            elements=ordered_elements,
            markdown=markdown,
            metadata={
                "total_elements": len(ordered_elements),
                "sources": list(set(elem.source for elem in ordered_elements)),
                "strategy": strategy
            }
        )
    
    def resolve_conflicts(self, elements: List[ContentElement]) -> List[ContentElement]:
        """Handle overlapping text/vision extractions"""
        resolved = []
        used_indices = set()
        
        for i, elem in enumerate(elements):
            if i in used_indices:
                continue
                
            # Find overlapping elements
            overlapping = [elem]
            for j, other in enumerate(elements[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if self._is_overlapping(elem, other):
                    overlapping.append(other)
                    used_indices.add(j)
            
            # Resolve the overlapping group
            best_element = self._select_best_element(overlapping)
            resolved.append(best_element)
            used_indices.add(i)
        
        return resolved
    
    def maintain_document_flow(self, elements: List[ContentElement]) -> List[ContentElement]:
        """Preserve logical document structure"""
        # Sort by reading order (top to bottom, left to right)
        def sort_key(elem):
            if elem.bbox:
                return (elem.bbox[1], elem.bbox[0])  # y, then x
            return elem.order
        
        sorted_elements = sorted(elements, key=sort_key)
        
        # Update order indices
        for i, elem in enumerate(sorted_elements):
            elem.order = i
        
        return sorted_elements
    
    def _convert_text_content(self, text_content: Any) -> List[ContentElement]:
        """Convert text extraction results to ContentElement format"""
        elements = []
        
        if hasattr(text_content, 'elements'):
            # ExtractedContent from TextExtractor
            for i, text_elem in enumerate(text_content.elements):
                content_type = "formula" if text_elem.is_formula else "text"
                
                element = ContentElement(
                    content=text_elem.text,
                    source=ContentSource.TEXT_EXTRACTION,
                    content_type=content_type,
                    confidence=0.9,  # High confidence for direct text
                    bbox=text_elem.bbox,
                    order=i
                )
                elements.append(element)
        
        elif isinstance(text_content, str):
            # Simple text string
            element = ContentElement(
                content=text_content,
                source=ContentSource.TEXT_EXTRACTION,
                content_type="text",
                confidence=0.8,
                order=0
            )
            elements.append(element)
        
        return elements
    
    def _convert_vision_content(self, vision_content: List[Any]) -> List[ContentElement]:
        """Convert vision model results to ContentElement format"""
        elements = []
        
        for i, vision_result in enumerate(vision_content):
            if hasattr(vision_result, 'content'):
                # VisionResult object
                element = ContentElement(
                    content=vision_result.content,
                    source=ContentSource.VISION_MODEL,
                    content_type=vision_result.content_type,
                    confidence=vision_result.confidence,
                    order=i
                )
                elements.append(element)
            elif isinstance(vision_result, dict):
                # Dictionary format
                element = ContentElement(
                    content=vision_result.get('content', ''),
                    source=ContentSource.VISION_MODEL,
                    content_type=vision_result.get('type', 'text'),
                    confidence=vision_result.get('confidence', 0.5),
                    order=i
                )
                elements.append(element)
        
        return elements
    
    def _merge_hybrid_content(self, text_elements: List[ContentElement], 
                            vision_elements: List[ContentElement]) -> List[ContentElement]:
        """Merge text and vision elements intelligently"""
        merged = []
        
        # Prioritize text for basic content, vision for complex content
        for text_elem in text_elements:
            if text_elem.content_type in ["text"] and len(text_elem.content.strip()) > 10:
                # Use text extraction for regular text
                merged.append(text_elem)
        
        for vision_elem in vision_elements:
            if vision_elem.content_type in ["table", "diagram", "formula"]:
                # Use vision for complex content
                merged.append(vision_elem)
            elif vision_elem.content_type == "text":
                # Only add vision text if no overlapping text extraction
                if not any(self._is_similar_content(vision_elem.content, t.content) 
                          for t in text_elements):
                    merged.append(vision_elem)
        
        return merged
    
    def _is_overlapping(self, elem1: ContentElement, elem2: ContentElement) -> bool:
        """Check if two elements overlap in content or position"""
        
        # Content similarity check
        if self._is_similar_content(elem1.content, elem2.content):
            return True
        
        # Spatial overlap check (if bboxes available)
        if elem1.bbox and elem2.bbox:
            return self._bbox_overlap(elem1.bbox, elem2.bbox) > 0.5
        
        return False
    
    def _is_similar_content(self, content1: str, content2: str) -> bool:
        """Check if two content strings are similar"""
        # Clean and normalize
        clean1 = re.sub(r'\s+', ' ', content1.strip().lower())
        clean2 = re.sub(r'\s+', ' ', content2.strip().lower())
        
        if not clean1 or not clean2:
            return False
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        return similarity > self.similarity_threshold
    
    def _bbox_overlap(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _select_best_element(self, overlapping: List[ContentElement]) -> ContentElement:
        """Select the best element from overlapping candidates"""
        if len(overlapping) == 1:
            return overlapping[0]
        
        # Score elements based on confidence and source preference
        scored = []
        for elem in overlapping:
            score = elem.confidence * self.confidence_bias[elem.source]
            
            # Bonus for specific content types
            if elem.content_type in ["table", "formula"]:
                score *= 1.2
            
            # Bonus for longer, more detailed content
            score *= min(len(elem.content) / 100, 1.5)
            
            scored.append((score, elem))
        
        # Return highest scored element
        return max(scored, key=lambda x: x[0])[1]
    
    def _elements_to_markdown(self, elements: List[ContentElement]) -> str:
        """Convert elements to final markdown format"""
        markdown_parts = []
        
        for elem in elements:
            content = elem.content.strip()
            if not content:
                continue
            
            if elem.content_type == "formula":
                # Wrap formulas in math notation
                if not content.startswith('$'):
                    content = f"${content}$"
            
            elif elem.content_type == "table":
                # Ensure proper table formatting
                if not content.startswith('|'):
                    # Try to format as table if not already
                    lines = content.split('\n')
                    if len(lines) > 1:
                        content = self._format_as_table(lines)
            
            elif elem.content_type == "diagram":
                # Add descriptive header
                content = f"**Diagram Description:**\n{content}"
            
            markdown_parts.append(content)
        
        return '\n\n'.join(markdown_parts)
    
    def _format_as_table(self, lines: List[str]) -> str:
        """Format text lines as markdown table"""
        if len(lines) < 2:
            return '\n'.join(lines)
        
        # Simple table formatting - split by common delimiters
        formatted_lines = []
        for line in lines:
            # Split by multiple spaces, tabs, or pipes
            cells = re.split(r'\s{2,}|\t+|\|+', line.strip())
            if len(cells) > 1:
                formatted_line = '| ' + ' | '.join(cell.strip() for cell in cells) + ' |'
                formatted_lines.append(formatted_line)
        
        # Add header separator after first line
        if len(formatted_lines) > 1:
            first_line_cells = formatted_lines[0].count('|') - 1
            separator = '|' + '---|' * first_line_cells
            formatted_lines.insert(1, separator)
        
        return '\n'.join(formatted_lines) if formatted_lines else '\n'.join(lines)