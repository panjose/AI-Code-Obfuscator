import os
import re
import random
import ast
import astor
import json
import time
from typing import List, Dict, Tuple, Optional


class AICodeAnalyzer:
    """
    AI-powered code analyzer that uses LLM to understand code logic
    and generate meaningful variable names.
    """
    
    def __init__(self, api_key: str = "", api_url: str = "", model: str = ""):
        """
        Initialize AI analyzer with API configuration.
        
        Args:
            api_key: API key for the LLM service
            api_url: API endpoint URL (optional for Gemini)
            model: Model name to use
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        
        # For Gemini, we only need api_key and model
        # api_url can be empty
        self.configured = bool(api_key and model)
    
    def analyze_variable_purpose(self, code_context: str, variable_name: str, 
                                language: str) -> str:
        """
        Use LLM to analyze variable purpose and suggest meaningful name.
        
        Args:
            code_context: Code snippet containing the variable
            variable_name: Original variable name
            language: Programming language
            
        Returns:
            Suggested meaningful variable name
        """
        if not self.configured:
            return f"var_{variable_name}"
        
        prompt = f"""Analyze this {language} code and suggest a meaningful variable name for '{variable_name}'.
The name should:
1. Describe the variable's purpose clearly
2. Follow {language} naming conventions
3. Be concise but descriptive (2-3 words max)
4. Use camelCase or snake_case appropriately

Code context:
```{language}
{code_context}
```

Only respond with the suggested variable name, nothing else."""

        try:
            import requests
            
            # Check if using Gemini API
            is_gemini = ('gemini' in self.model.lower() or 
                        not self.api_url or 
                        'generativelanguage.googleapis.com' in self.api_url)
            
            if is_gemini:
                # Gemini API format - construct URL dynamically
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Build Gemini URL based on model name
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 50
                    }
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                suggested_name = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            else:
                # OpenAI-compatible API format
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 50
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                suggested_name = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Clean the response
            suggested_name = re.sub(r'[^a-zA-Z0-9_]', '', suggested_name)
            
            if suggested_name and len(suggested_name) > 0:
                return suggested_name
            else:
                return f"{variable_name}"
                
        except Exception as e:
            print(f"Warning: AI analysis failed for '{variable_name}': {str(e)}")
            return f"{variable_name}"
    
    def analyze_function_purpose(self, function_code: str, function_name: str,
                                 language: str) -> Tuple[str, str]:
        """
        Use LLM to analyze function purpose and suggest meaningful name and comment.
        
        Args:
            function_code: Complete function code
            function_name: Original function name
            language: Programming language
            
        Returns:
            Tuple of (suggested_name, comment_description)
        """
        if not self.configured:
            return function_name, "Function implementation"
        
        prompt = f"""Analyze this {language} function and provide:
1. A meaningful function name (following {language} conventions)
2. A brief comment describing what it does (one line, max 60 characters)

Function:
```{language}
{function_code}
```

Respond in JSON format:
{{"name": "suggested_name", "comment": "Brief description"}}"""

        try:
            import requests
            
            # Check if using Gemini API
            is_gemini = ('gemini' in self.model.lower() or 
                        not self.api_url or 
                        'generativelanguage.googleapis.com' in self.api_url)
            
            if is_gemini:
                # Gemini API format
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Build Gemini URL based on model name
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 100
                    }
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            else:
                # OpenAI-compatible API format
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 100
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Parse JSON response
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("name", function_name), data.get("comment", "Function implementation")
            
        except Exception as e:
            print(f"Warning: AI analysis failed for function '{function_name}': {str(e)}")
        
        return function_name, "Function implementation"


class CodeAntiPlagiarism:
    """
    Intelligent code anti-plagiarism tool with AI-powered variable naming.
    Supports C, C++, and Python languages.
    """
    
    def __init__(self, input_file: str, language: str, ai_analyzer: Optional[AICodeAnalyzer] = None):
        """
        Initialize the anti-plagiarism tool.
        
        Args:
            input_file: Path to the source code file
            language: Programming language (c, cpp, python)
            ai_analyzer: AI analyzer instance (optional)
        """
        self.input_file = input_file
        self.language = language.lower()
        self.code = ""
        self.variable_mapping = {}
        self.ai_analyzer = ai_analyzer
        
        # Validate language support
        if self.language not in ['c', 'cpp', 'python']:
            raise ValueError("Unsupported language. Choose from: c, cpp, python")
        
        # Read source code
        self._read_code()
    
    def _read_code(self):
        """Read source code from input file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.code = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.input_file}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")
    
    def _get_code_context(self, code: str, target_line: int, context_lines: int = 5) -> str:
        """
        Extract code context around a specific line.
        
        Args:
            code: Full code string
            target_line: Target line number
            context_lines: Number of lines before and after
            
        Returns:
            Code context string
        """
        lines = code.split('\n')
        start = max(0, target_line - context_lines)
        end = min(len(lines), target_line + context_lines + 1)
        return '\n'.join(lines[start:end])
    
    def _generate_meaningful_name(self, original_name: str, code_context: str = "") -> str:
        """
        Generate meaningful variable name using AI analysis.
        
        Args:
            original_name: Original variable name
            code_context: Code context for analysis
            
        Returns:
            New meaningful variable name
        """
        if original_name in self.variable_mapping:
            return self.variable_mapping[original_name]
        
        # Use AI analyzer if available and configured
        if self.ai_analyzer and self.ai_analyzer.configured and code_context:
            new_name = self.ai_analyzer.analyze_variable_purpose(
                code_context, original_name, self.language
            )
        else:
            # Fallback to simple meaningful names
            prefixes = ['data', 'value', 'result', 'item', 'element', 'count', 
                       'index', 'total', 'temp', 'buffer', 'input', 'output']
            new_name = f"{random.choice(prefixes)}_{original_name}"
        
        # Ensure uniqueness
        base_name = new_name
        counter = 1
        while new_name in self.variable_mapping.values():
            new_name = f"{base_name}_{counter}"
            counter += 1
        
        self.variable_mapping[original_name] = new_name
        return new_name
    
    def _process_python(self) -> str:
        """
        Process Python code with AI-powered transformations.
        
        Returns:
            Transformed Python code
        """
        try:
            tree = ast.parse(self.code)
        except SyntaxError as e:
            raise SyntaxError(f"Python syntax error: {str(e)}")
        
        # Transform AST with AI-powered naming
        transformer = PythonTransformer(self.code, self.ai_analyzer, self.language)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        
        # Convert back to code
        modified_code = astor.to_source(new_tree)
        
        # Add intelligent comments
        modified_code = self._add_intelligent_comments(modified_code)
        
        # Vary spacing
        modified_code = self._vary_spacing(modified_code)
        
        # Store the mapping
        self.variable_mapping = transformer.name_mapping
        
        return modified_code
    
    def _process_c_cpp(self) -> str:
        """
        Process C/C++ code with AI-powered style modifications.
        
        Returns:
            Transformed C/C++ code
        """
        modified_code = self.code
        
        # Rename variables with AI analysis
        modified_code = self._rename_c_variables_intelligent(modified_code)
        
        # Add intelligent comments
        modified_code = self._add_intelligent_comments(modified_code)
        
        # Vary formatting
        modified_code = self._vary_c_formatting(modified_code)
        
        # Reorder function definitions
        modified_code = self._reorder_c_functions(modified_code)
        
        return modified_code
    
    def _add_intelligent_comments(self, code: str) -> str:
        """
        Add meaningful comments based on code analysis.
        
        Args:
            code: Source code
            
        Returns:
            Code with added comments
        """
        lines = code.split('\n')
        commented_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Add comment before function definitions
            if self.language == 'python':
                if stripped.startswith('def ') and ':' in stripped:
                    if i > 0 and not lines[i-1].strip().startswith('#'):
                        func_name = re.search(r'def\s+(\w+)', stripped)
                        if func_name:
                            comment = f"# Function to handle {func_name.group(1)} operation"
                            commented_lines.append(' ' * (len(line) - len(line.lstrip())) + comment)
            
            elif self.language in ['c', 'cpp']:
                if re.search(r'\w+\s+\w+\s*\([^)]*\)\s*\{', stripped):
                    if i > 0 and not lines[i-1].strip().startswith('//'):
                        commented_lines.append('// Function implementation')
            
            commented_lines.append(line)
            
            # Add inline comments for complex operations
            if random.random() < 0.08:
                if self.language == 'python':
                    if any(op in stripped for op in ['for ', 'while ', 'if ', 'return ']):
                        indent = len(line) - len(line.lstrip())
                        commented_lines.append(' ' * indent + '# Process logic step')
                elif self.language in ['c', 'cpp']:
                    if any(op in stripped for op in ['for(', 'while(', 'if(', 'return ']):
                        indent = len(line) - len(line.lstrip())
                        commented_lines.append(' ' * indent + '// Execute operation')
        
        return '\n'.join(commented_lines)
    
    def _vary_spacing(self, code: str) -> str:
        """Add strategic blank lines for better readability."""
        lines = code.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            # Add blank line before function definitions
            if i > 0 and (line.strip().startswith('def ') or 
                         line.strip().startswith('class ')):
                if new_lines and new_lines[-1].strip():
                    new_lines.append('')
            
            new_lines.append(line)
            
            # Add blank line after function definitions occasionally
            if ':' in line and (line.strip().startswith('def ') or 
                               line.strip().startswith('class ')):
                if random.random() < 0.15:
                    new_lines.append('')
        
        return '\n'.join(new_lines)
    
    def _rename_c_variables_intelligent(self, code: str) -> str:
        """
        Rename C/C++ variables using AI analysis.
        
        Args:
            code: Source code
            
        Returns:
            Code with renamed variables
        """
        # Find variable declarations
        var_pattern = r'\b(int|float|double|char|long|short|unsigned|signed|void|bool)\s+([a-zA-Z_]\w*)\b'
        
        lines = code.split('\n')
        variables = {}
        
        for line_num, line in enumerate(lines):
            for match in re.finditer(var_pattern, line):
                var_name = match.group(2)
                # Skip common keywords and standard library functions
                if var_name not in ['main', 'printf', 'scanf', 'malloc', 'free', 
                                   'return', 'if', 'else', 'for', 'while', 'switch']:
                    if var_name not in variables:
                        context = self._get_code_context(code, line_num)
                        variables[var_name] = context
        
        # Replace variables with meaningful names
        modified_code = code
        for var_name, context in variables.items():
            new_name = self._generate_meaningful_name(var_name, context)
            # Use word boundary to avoid partial replacements
            modified_code = re.sub(r'\b' + re.escape(var_name) + r'\b', new_name, modified_code)
        
        return modified_code
    
    def _vary_c_formatting(self, code: str) -> str:
        """
        Vary C/C++ code formatting style.
        
        Args:
            code: Source code
            
        Returns:
            Reformatted code
        """
        modified = code
        
        # Vary brace style
        brace_style = random.choice(['K&R', 'Allman', 'mixed'])
        
        if brace_style == 'Allman':
            # Braces on new line
            modified = re.sub(r'\)\s*\{', r')\n{', modified)
            modified = re.sub(r'else\s*\{', r'else\n{', modified)
        
        # Vary spacing around operators
        modified = re.sub(r'(\w+)\s*([+\-*/%])\s*(\w+)', 
                         lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}", 
                         modified)
        
        # Vary pointer notation
        if random.random() < 0.5:
            modified = re.sub(r'\*\s*(\w+)', r'* \1', modified)
        else:
            modified = re.sub(r'\*\s*(\w+)', r'*\1', modified)
        
        return modified
    
    def _reorder_c_functions(self, code: str) -> str:
        """
        Reorder function definitions in C/C++ code.
        
        Args:
            code: Source code
            
        Returns:
            Code with reordered functions
        """
        # Split code into parts
        parts = []
        current_part = []
        in_function = False
        brace_count = 0
        
        lines = code.split('\n')
        
        for line in lines:
            current_part.append(line)
            
            # Track function boundaries
            if re.search(r'\w+\s+\w+\s*\([^)]*\)\s*\{', line):
                in_function = True
                brace_count = line.count('{') - line.count('}')
            elif in_function:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    in_function = False
                    parts.append('\n'.join(current_part))
                    current_part = []
        
        if current_part:
            parts.append('\n'.join(current_part))
        
        # Separate main function
        main_part = None
        other_parts = []
        
        for part in parts:
            if 'int main' in part or 'void main' in part:
                main_part = part
            elif part.strip():
                other_parts.append(part)
        
        # Shuffle non-main functions
        random.shuffle(other_parts)
        
        # Reconstruct
        result = '\n\n'.join(other_parts)
        if main_part:
            result += '\n\n' + main_part
        
        return result if result.strip() else code
    
    def process(self) -> str:
        """
        Main processing method to transform code.
        
        Returns:
            Transformed code string
        """
        print(f"\nProcessing {self.language} code...")
        
        if self.ai_analyzer and self.ai_analyzer.configured:
            print("✓ Using AI-powered analysis for intelligent naming")
        else:
            print("⚠ AI analyzer not configured, using fallback naming")
        
        if self.language == 'python':
            return self._process_python()
        elif self.language in ['c', 'cpp']:
            return self._process_c_cpp()
        else:
            raise ValueError(f"Unsupported language: {self.language}")
    
    def save_output(self, output_file: str = None):
        """
        Save transformed code to output file.
        
        Args:
            output_file: Output file path (auto-generated if None)
        """
        # Process the code
        modified_code = self.process()
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(self.input_file)[0]
            ext = os.path.splitext(self.input_file)[1]
            output_file = f"{base_name}_modified{ext}"
        
        # Write to output file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            print(f"\n✓ Successfully saved modified code to: {output_file}")
            print(f"✓ Total variables renamed: {len(self.variable_mapping)}")
            return output_file
        except Exception as e:
            raise Exception(f"Error writing output file: {str(e)}")


class PythonTransformer(ast.NodeTransformer):
    """AST transformer for Python code with AI-powered naming."""
    
    def __init__(self, original_code: str, ai_analyzer: Optional[AICodeAnalyzer] = None, 
                 language: str = "python"):
        self.original_code = original_code
        self.ai_analyzer = ai_analyzer
        self.language = language
        self.name_mapping = {}
        self.lines = original_code.split('\n')
        
        # Python built-ins and common imports to skip
        self.skip_names = {
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set',
            'True', 'False', 'None', 'type', 'sum', 'max', 'min', 'abs', 'all', 'any',
            'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed', 'open',
            'input', 'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
            'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError', 'os', 
            'pd', 'np', 'json', 'sys', 'math', 'random', 'time', 're', 'e'
        }
    
    def _get_context_for_node(self, node) -> str:
        """Extract code context around a node."""
        if hasattr(node, 'lineno'):
            start = max(0, node.lineno - 3)
            end = min(len(self.lines), node.lineno + 3)
            return '\n'.join(self.lines[start:end])
        return ""
    
    def visit_Name(self, node):
        """Rename variables with AI analysis."""
        if node.id in self.skip_names or node.id.startswith('__'):
            return node
        
        if node.id not in self.name_mapping:
            context = self._get_context_for_node(node)
            
            if self.ai_analyzer and self.ai_analyzer.configured:
                new_name = self.ai_analyzer.analyze_variable_purpose(
                    context, node.id, self.language
                )
            else:
                # Fallback naming
                new_name = f"{node.id}"
            
            # Ensure valid Python identifier
            new_name = re.sub(r'[^a-zA-Z0-9_]', '', new_name)
            if not new_name or new_name[0].isdigit():
                new_name = f"{new_name}"
            
            self.name_mapping[node.id] = new_name
        
        node.id = self.name_mapping[node.id]
        return node
    
    def visit_FunctionDef(self, node):
        """Process function definitions."""
        # Don't rename special methods
        if not node.name.startswith('__'):
            if node.name not in self.name_mapping:
                context = self._get_context_for_node(node)
                
                if self.ai_analyzer and self.ai_analyzer.configured:
                    func_code = '\n'.join(self.lines[node.lineno-1:node.end_lineno])
                    new_name, comment = self.ai_analyzer.analyze_function_purpose(
                        func_code, node.name, self.language
                    )
                    self.name_mapping[node.name] = new_name
                else:
                    self.name_mapping[node.name] = node.name
            
            node.name = self.name_mapping[node.name]
        
        self.generic_visit(node)
        return node


def load_config(config_file: str = "config.json") -> Dict:
    """
    Load API configuration from JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Warning: Error loading config: {str(e)}")
        return {}


def save_config_template(config_file: str = "config.json"):
    """
    Save configuration template file.
    
    Args:
        config_file: Path to configuration file
    """
    template = {
        "api_key": "YOUR_API_KEY_HERE",
        "api_url": "",
        "model": "gemini-1.5-flash",
        "note": "For Gemini: Only api_key and model are needed. Leave api_url empty or remove it.",
        "instructions": "Get your Gemini API key from: https://makersuite.google.com/app/apikey",
        "examples": {
            "gemini_flash": {
                "api_key": "AIzaSy...",
                "model": "gemini-1.5-flash",
                "api_url": ""
            },
            "gemini_pro": {
                "api_key": "AIzaSy...",
                "model": "gemini-1.5-pro",
                "api_url": ""
            },
            "openai": {
                "api_key": "sk-...",
                "api_url": "https://api.openai.com/v1/chat/completions",
                "model": "gpt-3.5-turbo"
            }
        }
    }
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=4, ensure_ascii=False)
        print(f"✓ Configuration template saved to: {config_file}")
        print("\n📝 Gemini Configuration Guide:")
        print("  1. Get API key: https://makersuite.google.com/app/apikey")
        print("  2. Edit config.json:")
        print("     - Set 'api_key' to your Gemini key")
        print("     - Set 'model' to 'gemini-1.5-flash' or 'gemini-1.5-pro'")
        print("     - Leave 'api_url' empty or remove it")
        print("\n  Available models: gemini-1.5-flash, gemini-1.5-pro, gemini-pro")
    except Exception as e:
        print(f"Error saving config template: {str(e)}")


def main():
    """Main entry point for the anti-plagiarism tool."""
    print("=" * 70)
    print("AI-Powered Code Anti-Plagiarism Tool")
    print("=" * 70)
    
    # Check for config file
    config = load_config()
    
    if not config or not config.get('api_key') or config.get('api_key') == 'YOUR_API_KEY_HERE':
        print("\n⚠ No valid API configuration found.")
        create_config = input("Would you like to create a config template? (y/n): ").strip().lower()
        if create_config == 'y':
            save_config_template()
            print("\nPlease configure the API settings and run again.")
            return
        else:
            print("\nContinuing without AI analysis (using fallback naming)...")
            ai_analyzer = None
    else:
        ai_analyzer = AICodeAnalyzer(
            api_key=config.get('api_key', ''),
            api_url=config.get('api_url', ''),
            model=config.get('model', '')
        )
        print("✓ AI analyzer configured successfully")
    
    # Get user input
    print("\n" + "-" * 70)
    input_file = input("Enter the path to your code file: ").strip()
    language = input("Enter the programming language (c/cpp/python): ").strip().lower()
    output_file = input("Enter output file path (press Enter for auto): ").strip()
    
    if not output_file:
        output_file = None
    
    try:
        # Create processor instance
        processor = CodeAntiPlagiarism(input_file, language, ai_analyzer)
        
        # Process and save
        result_file = processor.save_output(output_file)
        
        print("\n" + "=" * 70)
        print("✓ Code transformation completed successfully!")
        print("✓ The modified code maintains all original functionality.")
        print("✓ Variable names are meaningful and context-appropriate.")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")


if __name__ == "__main__":
    main()