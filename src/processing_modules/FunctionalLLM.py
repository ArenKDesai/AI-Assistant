import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import inspect
import json
import sys
from jsonformer import Jsonformer

class FunctionalLLM:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.available_functions = {}
        
    def register_function(self, func):
        """Register a function and its metadata."""
        metadata = {
            'name': func.__name__,
            'description': func.__doc__,
            'parameters': inspect.signature(func).parameters
        }
        self.available_functions[func.__name__] = {
            'function': func,
            'metadata': metadata
        }
        
    def generate_function_context(self):
        """Create a string describing all available functions."""
        context = "Available functions:\n\n"
        for name, info in self.available_functions.items():
            context += f"Function: {name}\n"
            context += f"Description: {info['metadata']['description']}\n"
            context += f"Parameters: {info['metadata']['parameters']}\n\n"
        return context
        
    def process_prompt(self, user_prompt):
        """Process user prompt and decide which function to call."""
        # Combine function context with user prompt
        full_prompt = (
            self.generate_function_context() + 
            "\nUser request: " + user_prompt + 
            "\nResponse format: {'function': 'function_name', 'args': 'comma-delimited string array of arguments'}" + 
            """

Example:

Function: example_function
Description: This isn't a real function, but does describe what one may look like. 
Parameters: OrderedDict({'argument_one': <Parameter "argument_one: int">, 'arg2': <Parameter "arg2: float">, 'ParameterThree': <Parameter "ParameterThree:: str">})

output:
{'function': 'example_function', 'args': '32,12.95,example string'}

Example:

Function: example_function
Description: This isn't a real function, but does describe what one may look like. 
Parameters: OrderedDict({'test': <Parameter "test: bool">})

output:
{'function': 'example_function', 'args': 'True'}
            """
        )

        json_schema = {
            "type": "object",
            "properties": {
                "function": {"type": "string"},
                    "args": {"type": "string"}
            },
            "required": ["function", "args"]
        }

        jsonformer = Jsonformer(self.model, self.tokenizer, json_schema, full_prompt)
        generated_data = jsonformer()

        print ("\n FULL PROMPT")
        print(full_prompt)
        print("\n")
        print("\n RESPONSE")
        print(generated_data)
        
        try:
            # Parse the model's response to get function name and arguments
            func_name = generated_data['function']
            args = generated_data['args'].split(',')
            
            # Execute the chosen function
            if func_name in self.available_functions:
                result = self.available_functions[func_name]['function'](*args)
                return {
                    'success': True,
                    'function_called': func_name,
                    'result': result
                }
            else:
                return {
                    'success': False,
                    'error': f"Function {func_name} not found"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
def example_usage():
    # Define some sample functions
    def calculate_price(quantity: int, price: float):
        """Calculate total price including 10% tax."""
        quantity = int(quantity)
        price = float(price)
        return quantity * price * 1.1
    
    def format_address(street: str, city: str, country: str):
        """Format an address in a standard way."""
        return f"{street}\n{city}\n{country}"
    
    # Initialize and register functions
    llm = FunctionalLLM()
    llm.register_function(calculate_price)
    llm.register_function(format_address)
    
    # Process a user prompt
    result = llm.process_prompt("I need to calculate the price for 5 items that cost $10 each")
    print(result)
