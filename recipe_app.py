import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import pandas as pd
import os
import re
import random
from typing import List, Dict, Tuple, Optional

class RecipeGenerator:
    def _init_(self, model_name="pratultandon/recipe-nlg-gpt2"):
        """Initialize the recipe generator with a pre-trained model."""
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
    def generate_recipe(self, ingredients: List[str], cuisine_type: Optional[str] = None, 
                       cooking_time: Optional[str] = None, skill_level: Optional[str] = None) -> str:
        """Generate a recipe based on ingredients and optional parameters."""
        # Format the prompt
        prompt = "ingredients: " + ", ".join(ingredients)
        
        if cuisine_type:
            prompt += f" | cuisine: {cuisine_type}"
        if cooking_time:
            prompt += f" | time: {cooking_time}"
        if skill_level:
            prompt += f" | level: {skill_level}"
            
        prompt += " | recipe: "
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate recipe text
        output = self.model.generate(
            input_ids,
            max_length=300,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode the generated text
        recipe_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean up the generated text
        recipe_text = recipe_text.replace(prompt, "")
        
        # Format the recipe nicely
        return self._format_recipe(recipe_text)
    
    def _format_recipe(self, recipe_text: str) -> str:
        """Format the raw generated recipe text into a more readable format."""
        # Try to separate title, ingredients, and instructions
        parts = re.split(r'\n\s*\n', recipe_text, 1)
        
        if len(parts) > 1:
            title = parts[0].strip()
            rest = parts[1]
            
            # Try to separate ingredients and instructions
            match = re.search(r'(ingredients|what you need):(.*?)(instructions|directions|method|steps):', 
                             rest, re.IGNORECASE | re.DOTALL)
            
            if match:
                ingredients = match.group(2).strip()
                instructions = rest[match.end():].strip()
                
                # Format ingredients as a list
                ingredients_list = [f"• {item.strip()}" for item in ingredients.split('\n') if item.strip()]
                
                # Format instructions with numbered steps
                instructions_lines = instructions.split('\n')
                instructions_formatted = []
                step_num = 1
                
                for line in instructions_lines:
                    if line.strip():
                        # Remove existing numbers at the beginning
                        line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                        instructions_formatted.append(f"{step_num}. {line}")
                        step_num += 1
                
                # Build the formatted recipe
                formatted_recipe = f"# {title}\n\n"
                formatted_recipe += "## Ingredients\n"
                formatted_recipe += "\n".join(ingredients_list) + "\n\n"
                formatted_recipe += "## Instructions\n"
                formatted_recipe += "\n".join(instructions_formatted)
                
                return formatted_recipe
            
        # Fallback if we couldn't parse the recipe properly
        return recipe_text

class CulinaryAssistant:
    def _init_(self):
        """Initialize the culinary assistant with a recipe generator and pantry."""
        self.recipe_generator = RecipeGenerator()
        self.pantry = set()  # User's available ingredients
        self.cuisine_types = ["American", "Italian", "Mexican", "Chinese", "Indian", 
                             "Japanese", "French", "Mediterranean", "Thai", "Greek"]
        self.cooking_times = ["15 minutes", "30 minutes", "45 minutes", "1 hour", "2 hours"]
        self.skill_levels = ["beginner", "intermediate", "advanced"]
        
    def add_ingredient(self, ingredient: str) -> None:
        """Add an ingredient to the user's pantry."""
        if ingredient.strip():
            self.pantry.add(ingredient.lower().strip())
            
    def remove_ingredient(self, ingredient: str) -> None:
        """Remove an ingredient from the user's pantry."""
        if ingredient in self.pantry:
            self.pantry.remove(ingredient)
            
    def clear_pantry(self) -> None:
        """Clear all ingredients from the pantry."""
        self.pantry.clear()
        
    def get_recipe(self, cuisine_type: Optional[str] = None, 
                  cooking_time: Optional[str] = None,
                  skill_level: Optional[str] = None) -> str:
        """Generate a recipe based on pantry ingredients and preferences."""
        if not self.pantry:
            return "Please add some ingredients to your pantry first."
        
        return self.recipe_generator.generate_recipe(
            list(self.pantry), cuisine_type, cooking_time, skill_level
        )
        
    def get_suggested_ingredients(self) -> List[str]:
        """Return a list of commonly used ingredients as suggestions."""
        common_ingredients = [
            "chicken", "beef", "pork", "salmon", "shrimp",
            "rice", "pasta", "potatoes", "bread", "flour",
            "tomatoes", "onions", "garlic", "bell peppers", "carrots",
            "cheese", "milk", "eggs", "butter", "olive oil",
            "basil", "oregano", "thyme", "rosemary", "cumin"
        ]
        return common_ingredients

# Streamlit interface
def main():
    st.set_page_config(page_title="Smart Culinary Assistant", page_icon="🍳")
    
    st.title("🍳 Smart Culinary Assistant")
    st.write("Get personalized recipe recommendations based on your available ingredients.")
    
    # Initialize the assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = CulinaryAssistant()
    
    # Sidebar for ingredient management
    st.sidebar.header("Your Pantry 🧺")
    
    # Ingredient input
    new_ingredient = st.sidebar.text_input("Add an ingredient:")
    add_button = st.sidebar.button("Add to Pantry")
    
    if add_button and new_ingredient:
        st.session_state.assistant.add_ingredient(new_ingredient)
        st.sidebar.success(f"Added {new_ingredient} to your pantry!")
    
    # Suggested ingredients
    st.sidebar.subheader("Suggested Ingredients")
    suggestions = st.session_state.assistant.get_suggested_ingredients()
    
    # Display suggestions in a 5-column grid
    cols = st.sidebar.columns(2)
    for i, suggestion in enumerate(random.sample(suggestions, min(10, len(suggestions)))):
        if cols[i % 2].button(suggestion, key=f"sug_{suggestion}"):
            st.session_state.assistant.add_ingredient(suggestion)
            st.sidebar.success(f"Added {suggestion} to your pantry!")
    
    # Display the current pantry
    st.sidebar.subheader("Current Ingredients")
    
    if st.session_state.assistant.pantry:
        for ingredient in sorted(st.session_state.assistant.pantry):
            cols = st.sidebar.columns([3, 1])
            cols[0].write(ingredient)
            if cols[1].button("Remove", key=f"rem_{ingredient}"):
                st.session_state.assistant.remove_ingredient(ingredient)
                st.experimental_rerun()
        
        if st.sidebar.button("Clear All Ingredients"):
            st.session_state.assistant.clear_pantry()
            st.experimental_rerun()
    else:
        st.sidebar.write("Your pantry is empty. Add some ingredients!")
    
    # Main section for recipe generation
    st.subheader("Recipe Generator")
    
    col1, col2 = st.columns(2)
    
    # Recipe preferences
    with col1:
        cuisine_type = st.selectbox(
            "Cuisine Type", 
            ["Any"] + st.session_state.assistant.cuisine_types
        )
        
    with col2:
        cooking_time = st.selectbox(
            "Cooking Time", 
            ["Any"] + st.session_state.assistant.cooking_times
        )
    
    skill_level = st.selectbox(
        "Skill Level", 
        ["Any"] + st.session_state.assistant.skill_levels
    )
    
    # Generate button
    if st.button("Generate Recipe", type="primary"):
        if not st.session_state.assistant.pantry:
            st.error("Please add some ingredients to your pantry first.")
        else:
            with st.spinner("Generating your personalized recipe..."):
                cuisine = None if cuisine_type == "Any" else cuisine_type
                time = None if cooking_time == "Any" else cooking_time
                skill = None if skill_level == "Any" else skill_level
                
                recipe = st.session_state.assistant.get_recipe(cuisine, time, skill)
                
                st.markdown(recipe)
    
    # App information
    st.markdown("---")
    st.markdown("""
    *About this app:*
    
    This Smart Culinary Assistant uses a fine-tuned language model trained on the RecipeNLG dataset to generate personalized 
    recipes based on your available ingredients and preferences. Unlike traditional recipe apps that rely on static databases, 
    this assistant dynamically generates unique recipes tailored to your specific needs.
    
    *How to use:*
    1. Add ingredients to your pantry using the sidebar
    2. Select your preferred cuisine type, cooking time, and skill level
    3. Click "Generate Recipe" to get a personalized recipe
    
    Note: The generated recipes are AI-created and may require adjustments.
    """)

if __name__ == "_main_":
    main()