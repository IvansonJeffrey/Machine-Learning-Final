# Machine-Learning-Clothing

Dataset from: https://drive.google.com/drive/folders/1An2c_ZCkeGmhJg0zUjtZF46vyJgQwIr2 

<img width="369" height="442" alt="image" src="https://github.com/user-attachments/assets/3a293041-3aa8-4908-899b-dfa00babc3a6" />

## Setup

1. Make sure to create a folder called `checkpoints`, and put all the `.pth` files inside
2. Install required packages:
   ```bash
   pip install torch torchvision pillow numpy tqdm
   ```
3. For style evaluation (optional):
   ```bash
   pip install openai python-dotenv
   ```
4. Set up your OpenAI API key:
   - Edit the `.env` file and replace `your-api-key-here` with your actual API key
   - Get your API key from: https://platform.openai.com/api-keys

## Usage

### Basic Inference (without style evaluation)

```bash
python output.py --image path/to/image.jpg
```

This will generate:
- Clothing categories (e.g., "grey jacket", "purple pants", "orange shoes")
- Detailed attributes (sleeve length, fabric, colors, etc.)
- Region-based color analysis

### With ChatGPT Style Evaluation

1. Make sure you've installed the packages and set up your `.env` file (see Setup above)

2. Run with style evaluation:
   ```bash
   python output.py --image path/to/image.jpg --evaluate_style
   ```

   The script will automatically load your API key from the `.env` file.

   **Alternative:** You can also pass the API key directly:
   ```bash
   python output.py --image path/to/image.jpg --evaluate_style --openai_key "your-api-key-here"
   ```
   
   Or set it as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   python output.py --image path/to/image.jpg --evaluate_style
   ```

The output will include a `style_evaluation` section with:
- Rating: "excellent", "good", "fair", or "poor"
- Score: 1-10
- Feedback: Brief explanation of what works/doesn't work
- Suggestions: Specific improvement recommendations

## Output Format

The `output.txt` file will contain:
```json
{
  "categories": ["grey jacket", "light grey shirt", "purple pants", "orange shoes"],
  "attributes": {...},
  "region_colors_parsing": {...},
  "style_evaluation": {
    "rating": "good",
    "score": 7,
    "feedback": "...",
    "suggestions": [...]
  }
}
```
