# PDF to Markdown Pipeline Documentation

## Installation

```bash
# Required dependencies
pip install PyMuPDF langchain-ollama pillow pydantic

# Install and run Ollama with vision model
ollama pull llama3.2-vision:11b
ollama serve
```

## Quick Start

### Simple Usage
```python
from pdf_markdown_pipeline import convert_pdf_to_markdown

# Basic conversion
result = convert_pdf_to_markdown(
    pdf_path="document.pdf",
    ollama_model="llama3.2-vision:11b",
    ollama_base_url="http://localhost:11434",
    output_dir="./output"
)

if result.success:
    print(f"Converted {len(result.pages)} pages")
else:
    print("Errors:", result.errors)
```

### Advanced Usage
```python
from pdf_markdown_pipeline import PDFToMarkdownPipeline, PipelineConfig

# Custom configuration
config = PipelineConfig()
config.dpi = 400                    # Higher resolution
config.vision_model_temp = 0.1      # Lower temperature for consistency
config.preserve_formatting = True   # Maintain formatting
config.image_embed_mode = "base64"  # Embed images as base64

# Initialize pipeline
pipeline = PDFToMarkdownPipeline(
    ollama_model="llama3.2-vision:11b",
    ollama_base_url="http://localhost:11434",
    config=config
)

# Convert PDF
result = pipeline.convert_pdf("document.pdf")
saved_files = pipeline.save_results(result, "./output")
```

## Configuration Options

### PipelineConfig Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dpi` | 300 | Image resolution for vision processing |
| `vision_model_temp` | 0.1 | Temperature for vision model (0.0-1.0) |
| `text_extraction_priority` | True | Prefer text extraction when possible |
| `preserve_formatting` | True | Maintain original formatting |
| `image_embed_mode` | "base64" | Image embedding: "base64" or "file" |

### Processing Strategies
- **TEXT_ONLY**: Direct text extraction for simple documents
- **VISION_ONLY**: Vision model for scanned/complex documents  
- **HYBRID**: Combines text + vision for optimal results

## API Reference

### Main Functions

#### `convert_pdf_to_markdown(pdf_path, ollama_model, ollama_base_url, output_dir)`
Simple conversion function with automatic file saving.

**Parameters:**
- `pdf_path` (str): Path to PDF file
- `ollama_model` (str): Ollama model name
- `ollama_base_url` (str): Ollama server URL
- `output_dir` (str): Output directory

**Returns:** `ConversionResult`

#### `PDFToMarkdownPipeline(ollama_model, ollama_base_url, config)`
Main pipeline class for advanced usage.

**Methods:**
- `convert_pdf(pdf_path)`: Convert entire PDF
- `convert_page(page)`: Convert single page
- `save_results(result, output_dir)`: Save to files
- `get_pipeline_info()`: Get configuration info

### Content-Specific Extraction

```python
# Extract specific content types
page_image = Utils.extract_page_image(page, dpi=300)

# Tables
table_result = vision_processor.extract_table_data(page_image)

# Mathematical formulas  
formula_result = vision_processor.extract_formulas(page_image)

# Diagrams/charts
diagram_result = vision_processor.describe_diagrams(page_image)
```

## Output Format

### File Structure
```
output/
├── page_001.md          # Individual pages
├── page_002.md
├── combined_document.md # All pages combined
└── conversion_metadata.json
```

### Markdown Features
- **Headers**: Detected from font sizes
- **Tables**: Proper markdown table format
- **Formulas**: LaTeX notation (`$inline$`, `$$display$$`)
- **Lists**: Bullet and numbered lists
- **Images**: Base64 embedded or file references
- **Diagrams**: Descriptive text sections

## Error Handling

### Common Issues
1. **Ollama Connection**: Check server URL and model availability
2. **PDF Access**: Verify file permissions and path
3. **Memory**: Large PDFs may need processing in batches
4. **Vision Model**: Ensure model supports multi-modal input

### Debugging
```python
# Check pipeline configuration
info = pipeline.get_pipeline_info()
print(info)

# Analyze page before processing
analysis = pipeline.analyzer.analyze_page_content(page)
print(f"Strategy: {analysis.strategy.value}")
print(f"Confidence: {analysis.confidence}")

# Process single page for testing
page_markdown = pipeline.convert_page(page)
```

## Performance Tips

### Optimization
- Use `text_extraction_priority=True` for text-heavy documents
- Lower `dpi` (200-300) for faster processing
- Use `vision_model_temp=0.1` for consistent output
- Process in batches for large documents

### Model Selection
- **llama3.2-vision:11b**: Good balance of speed/quality
- **llava:13b**: Alternative vision model
- **bakllava**: Specialized for document analysis

## Examples

### Mathematical Documents
```python
config = PipelineConfig()
config.vision_model_temp = 0.05  # Very consistent for formulas
config.preserve_formatting = True

pipeline = PDFToMarkdownPipeline("llama3.2-vision:11b", "http://localhost:11434", config)
```

### Technical Diagrams
```python
config = PipelineConfig() 
config.dpi = 400  # Higher resolution for detailed diagrams
config.image_embed_mode = "file"  # Save diagrams as separate files

pipeline = PDFToMarkdownPipeline("llama3.2-vision:11b", "http://localhost:11434", config)
```

### Scanned Documents
```python
config = PipelineConfig()
config.text_extraction_priority = False  # Force vision processing
config.dpi = 400  # Higher resolution for OCR

pipeline = PDFToMarkdownPipeline("llama3.2-vision:11b", "http://localhost:11434", config)
```

## Troubleshooting

### Connection Issues
```python
# Test Ollama connection
from langchain_ollama import ChatOllama
try:
    chat = ChatOllama(model="llama3.2-vision:11b", base_url="http://localhost:11434")
    print("✅ Connection successful")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Model Availability
```bash
# Check available models
ollama list

# Pull missing model
ollama pull llama3.2-vision:11b
```

### Memory Issues
- Process pages individually for large PDFs
- Reduce DPI setting
- Use smaller vision models if available
- Close PDF documents after processing
