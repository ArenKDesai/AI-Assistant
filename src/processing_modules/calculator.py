from FunctionalLLM import FunctionalLLM

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

def multiply()
