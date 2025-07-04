import re
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ImageReference:
    """Reference to an embedded image"""
    filename: str
    base64_data: str
    alt_text: str

class MarkdownGenerator:
    """Generate high-fidelity markdown output"""
    
    def __init__(self, config):
        self.config = config
        self.image_counter = 0
        self.embedded_images = []
    
    def format_mathematical_content(self, formulas: List[str]) -> str:
        """Convert to LaTeX notation in markdown"""
        formatted_formulas = []
        
        for formula in formulas:
            formula = formula.strip()
            if not formula:
                continue
            
            # Detect if it's inline or display math
            if self._is_display_formula(formula):
                # Display math (block)
                if not formula.startswith('$'):
                    formula = f"${formula}$"
            else:
                # Inline math
                if not formula.startswith("$"):
                    formula = f"${formula}$"
            
            formatted_formulas.append(formula)
        
        return '\n\n'.join(formatted_formulas)
    
    def structure_tables(self, table_data: List[str]) -> str:
        """Create properly formatted markdown tables"""
        formatted_tables = []
        
        for table in table_data:
            formatted_table = self._format_single_table(table)
            if formatted_table:
                formatted_tables.append(formatted_table)
        
        return '\n\n'.join(formatted_tables)
    
    def embed_images(self, images: List[Dict], mode: str = "base64") -> str:
        """Handle image embedding (base64/file refs)"""
        image_markdown = []
        
        for img_data in images:
            if mode == "base64":
                markdown = self._embed_base64_image(img_data)
            else:  # file mode
                markdown = self._embed_file_image(img_data)
            
            if markdown:
                image_markdown.append(markdown)
        
        return '\n\n'.join(image_markdown)
    
    def generate_final_markdown(self, processed_content: Dict) -> str:
        """Combine all elements into coherent markdown"""
        sections = []
        
        # Add title if available
        if 'title' in processed_content:
            sections.append(f"# {processed_content['title']}")
        
        # Process content by type
        if 'text' in processed_content:
            sections.append(self._format_text_content(processed_content['text']))
        
        if 'tables' in processed_content:
            sections.append(self.structure_tables(processed_content['tables']))
        
        if 'formulas' in processed_content:
            sections.append(self.format_mathematical_content(processed_content['formulas']))
        
        if 'images' in processed_content:
            sections.append(self.embed_images(processed_content['images'], self.config.image_embed_mode))
        
        if 'diagrams' in processed_content:
            sections.append(self._format_diagrams(processed_content['diagrams']))
        
        # Combine sections
        markdown = '\n\n'.join(filter(None, sections))
        
        # Post-process for cleanup
        markdown = self._clean_markdown(markdown)
        
        return markdown
    
    def _is_display_formula(self, formula: str) -> bool:
        """Determine if formula should be display (block) math"""
        display_indicators = [
            r'\\begin\{',  # LaTeX environments
            r'\\sum_',     # Summations
            r'\\int_',     # Integrals
            r'\\prod_',    # Products
            r'\\frac\{.*\}\{.*\}',  # Fractions
            r'=.*[+\-].*=',  # Multi-step equations
        ]
        
        return any(re.search(pattern, formula) for pattern in display_indicators)
    
    def _format_single_table(self, table_data: str) -> str:
        """Format a single table into proper markdown"""
        lines = table_data.strip().split('\n')
        if len(lines) < 2:
            return table_data
        
        # Clean and align table
        cleaned_lines = []
        max_cols = 0
        
        for line in lines:
            # Split by pipes and clean
            if '|' in line:
                cols = [col.strip() for col in line.split('|')]
                # Remove empty first/last if pipes at start/end
                if cols and not cols[0]:
                    cols = cols[1:]
                if cols and not cols[-1]:
                    cols = cols[:-1]
            else:
                # Split by multiple spaces/tabs
                cols = re.split(r'\s{2,}|\t+', line.strip())
            
            if cols:
                cleaned_lines.append(cols)
                max_cols = max(max_cols, len(cols))
        
        if not cleaned_lines:
            return table_data
        
        # Normalize column count
        for line in cleaned_lines:
            while len(line) < max_cols:
                line.append('')
        
        # Format as markdown table
        formatted_lines = []
        for i, line in enumerate(cleaned_lines):
            formatted_line = '| ' + ' | '.join(line) + ' |'
            formatted_lines.append(formatted_line)
            
            # Add header separator after first line
            if i == 0:
                separator = '| ' + ' | '.join(['---'] * len(line)) + ' |'
                formatted_lines.append(separator)
        
        return '\n'.join(formatted_lines)
    
    def _embed_base64_image(self, img_data: Dict) -> str:
        """Embed image as base64 data URL"""
        self.image_counter += 1
        
        base64_data = img_data.get('base64', '')
        alt_text = img_data.get('alt', f'Image {self.image_counter}')
        title = img_data.get('title', '')
        
        if not base64_data:
            return f"![{alt_text}](image_missing)"
        
        # Create data URL
        mime_type = img_data.get('mime_type', 'image/png')
        data_url = f"data:{mime_type};base64,{base64_data}"
        
        if title:
            return f'![{alt_text}]({data_url} "{title}")'
        else:
            return f'![{alt_text}]({data_url})'
    
    def _embed_file_image(self, img_data: Dict) -> str:
        """Embed image as file reference"""
        self.image_counter += 1
        
        filename = img_data.get('filename', f'image_{self.image_counter}.png')
        alt_text = img_data.get('alt', f'Image {self.image_counter}')
        title = img_data.get('title', '')
        
        # Save base64 data to file if provided
        if 'base64' in img_data:
            self._save_image_file(filename, img_data['base64'])
        
        if title:
            return f'![{alt_text}]({filename} "{title}")'
        else:
            return f'![{alt_text}]({filename})'
    
    def _save_image_file(self, filename: str, base64_data: str) -> None:
        """Save base64 image data to file"""
        try:
            image_data = base64.b64decode(base64_data)
            with open(filename, 'wb') as f:
                f.write(image_data)
        except Exception as e:
            print(f"Warning: Could not save image {filename}: {e}")
    
    def _format_text_content(self, text_content: str) -> str:
        """Format regular text content with proper markdown"""
        if not text_content:
            return ""
        
        # Split into paragraphs
        paragraphs = text_content.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect and format lists
            if self._is_list_paragraph(para):
                formatted_para = self._format_list(para)
            else:
                formatted_para = para
            
            formatted_paragraphs.append(formatted_para)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _format_diagrams(self, diagrams: List[str]) -> str:
        """Format diagram descriptions"""
        formatted_diagrams = []
        
        for i, diagram in enumerate(diagrams, 1):
            if diagram.strip():
                formatted = f"### Diagram {i}\n\n{diagram}"
                formatted_diagrams.append(formatted)
        
        return '\n\n'.join(formatted_diagrams)
    
    def _is_list_paragraph(self, paragraph: str) -> bool:
        """Check if paragraph contains list items"""
        lines = paragraph.split('\n')
        list_lines = 0
        
        for line in lines:
            if re.match(r'^\s*[•·◦▪▫\-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                list_lines += 1
        
        return list_lines > 1 or (len(lines) == 1 and list_lines == 1)
    
    def _format_list(self, paragraph: str) -> str:
        """Format list items properly"""
        lines = paragraph.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Bullet lists
            if re.match(r'^[•·◦▪▫]\s+', line):
                formatted_line = re.sub(r'^[•·◦▪▫]\s+', '- ', line)
                formatted_lines.append(formatted_line)
            # Numbered lists
            elif re.match(r'^\d+\.\s+', line):
                formatted_lines.append(line)
            # Dash/asterisk lists
            elif re.match(r'^[-*+]\s+', line):
                formatted_lines.append(line)
            else:
                # Regular line, might be continuation
                if formatted_lines:
                    formatted_lines.append(f"  {line}")  # Indent continuation
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _clean_markdown(self, markdown: str) -> str:
        """Clean up and optimize markdown output"""
        # Remove excessive whitespace
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Clean up table formatting
        markdown = re.sub(r'\|\s*\|', '| |', markdown)  # Fix empty cells
        
        # Ensure proper spacing around headers
        markdown = re.sub(r'(\n)(#{1,6}\s)', r'\1\n\2', markdown)
        
        # Fix formula spacing
        markdown = re.sub(r'\$\$\s*\$\ ', '', markdown)  # Remove empty math blocks
        
        # Clean up list formatting
        markdown = re.sub(r'\n(\s*[-*+]\s)', r'\n\n\1', markdown)  # Space before lists
        
        return markdown.strip()