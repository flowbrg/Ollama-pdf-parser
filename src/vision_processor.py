from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from src.logger_config import get_logger

@dataclass
class VisionResult:
    """Result from vision model processing"""
    content: str
    content_type: str
    confidence: float
    metadata: Dict = None

class VisionProcessor:
    """Use ChatOllama for complex content (diagrams, tables, handwritten)"""
    
    def __init__(self, model_name: str, base_url: str, temperature: float = 0.1):
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.chat_model = self._init_ollama(model_name, base_url)
        
        # Prompts for different content types
        self.prompts = {
            "table": self._get_table_prompt(),
            "diagram": self._get_diagram_prompt(),
            "formula": self._get_formula_prompt(),
            "general": self._get_general_prompt()
        }
        
        self.logger.info(f"VisionProcessor initialized with model: {model_name}")
    
    def _init_ollama(self, model_name: str, base_url: str) -> ChatOllama:
        """Initialize ChatOllama with custom base_url"""
        try:
            chat_model = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=self.temperature
            )
            self.logger.info(f"Connected to Ollama at {base_url}")
            return chat_model
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatOllama: {e}")
            raise ConnectionError(f"Failed to initialize ChatOllama: {e}")
    
    def process_image_content(self, image_base64: str, content_type: str = "general", 
                            context: str = "") -> VisionResult:
        """Process images/diagrams with context-aware prompts"""
        
        self.logger.debug(f"Processing image content with type: {content_type}")
        
        prompt = self.prompts.get(content_type, self.prompts["general"])
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        try:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
            
            message = HumanMessage(content=content)
            response = self.chat_model.invoke([message])
            
            self.logger.debug(f"Vision model response received, length: {len(response.content)}")
            
            return VisionResult(
                content=response.content,
                content_type=content_type,
                confidence=0.8,  # Default confidence
                metadata={"model": self.model_name, "prompt_type": content_type}
            )
            
        except Exception as e:
            self.logger.error(f"Vision processing failed: {e}")
            raise RuntimeError(f"Vision processing failed: {e}")
    
    def extract_table_data(self, image_base64: str) -> VisionResult:
        """Extract tables with structure preservation"""
        self.logger.debug("Extracting table data from image")
        return self.process_image_content(image_base64, "table")
    
    def describe_diagrams(self, image_base64: str, diagram_type: str = "") -> VisionResult:
        """Generate detailed diagram descriptions"""
        self.logger.debug(f"Describing diagram: {diagram_type}")
        context = f"This appears to be a {diagram_type} diagram." if diagram_type else ""
        return self.process_image_content(image_base64, "diagram", context)
    
    def extract_formulas(self, image_base64: str) -> VisionResult:
        """Extract mathematical formulas in LaTeX format"""
        self.logger.debug("Extracting formulas from image")
        return self.process_image_content(image_base64, "formula")
    
    def _get_table_prompt(self) -> str:
        """Prompt for table extraction"""
        return """Extract the table data from this image and format it as a markdown table.

Instructions:
1. Preserve the exact structure and all content
2. Use proper markdown table syntax with | separators
3. Include headers if present
4. Maintain data alignment and formatting
5. If cells are merged, indicate this clearly
6. If text is unclear, use [unclear] placeholder

Output only the markdown table, no additional text."""

    def _get_diagram_prompt(self) -> str:
        """Prompt for diagram description"""
        return """Analyze this diagram/chart/figure and provide a detailed description.

Instructions:
1. Identify the type of diagram (flowchart, network, schema, etc.)
2. Describe the main components and their relationships
3. Include any labels, arrows, or connections
4. Mention colors, shapes, or visual elements that convey meaning
5. Explain the overall purpose or message of the diagram
6. Use clear, structured language

Format as markdown with appropriate headers and lists."""

    def _get_formula_prompt(self) -> str:
        """Prompt for mathematical formula extraction"""
        return """Extract all mathematical formulas, equations, and expressions from this image.

Instructions:
1. Convert to proper LaTeX notation
2. Use $...$ for inline math and $$...$$ for display math
3. Preserve exact mathematical meaning
4. Include variable definitions if shown
5. Maintain equation numbering if present
6. If handwritten, interpret carefully and note uncertainty

Output only the mathematical content in LaTeX format."""

    def _get_general_prompt(self) -> str:
        """General prompt for mixed content"""
        return """Extract and describe all content from this image accurately.

Instructions:
1. Identify and extract all text content
2. Describe any images, diagrams, or charts
3. Convert tables to markdown format
4. Convert mathematical formulas to LaTeX
5. Preserve formatting and structure
6. Use appropriate markdown formatting

Provide complete, accurate extraction maintaining the original meaning and structure."""

    def batch_process(self, images_data: List[Tuple[str, str]]) -> List[VisionResult]:
        """Process multiple images with their content types"""
        self.logger.info(f"Batch processing {len(images_data)} images")
        results = []
        
        for i, (image_base64, content_type) in enumerate(images_data):
            try:
                self.logger.debug(f"Processing image {i+1}/{len(images_data)} of type {content_type}")
                result = self.process_image_content(image_base64, content_type)
                results.append(result)
            except Exception as e:
                # Continue processing other images even if one fails
                self.logger.warning(f"Failed to process image {i+1}: {e}")
                error_result = VisionResult(
                    content=f"Error processing image: {e}",
                    content_type=content_type,
                    confidence=0.0,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        return results
    
    def validate_extraction(self, result: VisionResult, expected_type: str) -> bool:
        """Basic validation of extraction results"""
        if result.content_type != expected_type:
            return False
        
        if expected_type == "table":
            # Check if output contains markdown table syntax
            return "|" in result.content and "-" in result.content
        
        elif expected_type == "formula":
            # Check if output contains LaTeX syntax
            return "$" in result.content or "\\[" in result.content
        
        elif expected_type == "diagram":
            # Check if output is descriptive (minimum length)
            return len(result.content.strip()) > 50
        
        return True  # General content always passes