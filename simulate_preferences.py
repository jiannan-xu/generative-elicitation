from openai import OpenAI
import json
import random

client = OpenAI(
    api_key = "",
)

def generate_random_customer_profile():
    # Define some random attributes for customer profiles
    current_cards_options = [
        "Chase Sapphire Preferred", "American Express Blue Cash Everyday", 
        "Capital One Quicksilver", "Citi Double Cash", "Wells Fargo Active Cash", 
        "No Credit Card"
    ]
    current_cards = random.choice(current_cards_options)
    
    # Apply the hard rule for account age
    if current_cards == "No Credit Card":
        account_age = "0 years"
        chase_5_24_statuses = ["0/24"]
    else:
        account_age = f"{random.randint(1, 20)} years"
        chase_5_24_statuses = [f"{random.randint(0, 5)}/24" for _ in range(100)]
    
    fico_scores = random.randint(600, 850)
    incomes = [f"${random.randint(10000, 150000)}" for _ in range(100)]
    monthly_spends = {
        "dining": f"${random.randint(50, 500)}",
        "groceries": f"${random.randint(100, 800)}",
        "gas": f"${random.randint(50, 300)}",
        "travel": f"${random.randint(0, 1000)}",
        "other": f"${random.randint(100, 1000)} (misc: shopping, healthcare, pet care, gym, etc)"
    }
    open_to_business_cards = random.choice(["Yes", "No"])
    next_card_purpose = random.choice(["Maximizing travel rewards", "Lower interest rate for balance transfer", "Cashback on everyday spending", "Building credit history", "Business expenses"])
    looking_at_cards = random.choice(["Chase Sapphire Reserve", "Capital One Venture", "Citi Simplicity", "Wells Fargo Reflect", "American Express Gold"])
    category_spending_preference = random.choice(["Open to category spending", "General spending card", "No preference", "Travel rewards card", "Cashback card", "Business card", "Student card"])

    profile = {
        "Current_cards": current_cards,
        "FICO_Score": fico_scores,
        "Oldest_account_age": account_age,
        "Chase_5_24_status": random.choice(chase_5_24_statuses),
        "Income": random.choice(incomes),
        "Average_monthly_spend": monthly_spends,
        "Open_to_Business_Cards": open_to_business_cards,
        "Next_card_purpose": next_card_purpose,
        "Looking_at_cards": looking_at_cards,
        "Category_spending_preference": category_spending_preference
    }
    return profile

def simulate_preferences(profile):
    # Create a prompt for the LLM based on the detailed customer profile
    prompt = f"""
    You are a customer with the following profile:
    - Current cards: {profile['Current_cards']}
    - FICO Score: {profile['FICO_Score']}
    - Oldest account age: {profile['Oldest_account_age']}
    - Chase 5/24 status: {profile['Chase_5_24_status']}
    - Income: {profile['Income']}
    - Average monthly spend:
        - Dining: {profile['Average_monthly_spend']['dining']}
        - Groceries: {profile['Average_monthly_spend']['groceries']}
        - Gas: {profile['Average_monthly_spend']['gas']}
        - Travel: {profile['Average_monthly_spend']['travel']}
        - Other: {profile['Average_monthly_spend']['other']}
    - Open to Business Cards: {profile['Open_to_Business_Cards']}
    - Purpose of your next card: {profile['Next_card_purpose']}
    - Cards you are considering: {profile['Looking_at_cards']}
    - Preference for category spending or general spending card: {profile['Category_spending_preference']}
    
    Based on this profile, please describe the type of credit card features and benefits you would prefer. Provide reasons for your choices, and mention if the cards you are considering are suitable for your needs.
    """

    # Generate a response using the LLM

    response = client.chat.completions.create( # Change the method
    model = "gpt-4-turbo",
    messages = [ # Change the prompt parameter to messages parameter
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},])
    return response.choices[0].message.content

def main():
    # List to store all simulated profiles and their preferences
    profiles_with_preferences = []

    # set random seed for reproducibility
    random.seed(42)

    # Simulate 100 customer profiles
    for i in range(100):
        profile = generate_random_customer_profile()
        preferences = simulate_preferences(profile)
        profiles_with_preferences.append({
            "profile": profile,
            "preferences": preferences
        })
        print(f"Simulated customer profile {i+1}.")


    # Save the results to a JSON file
    with open('simulated_customer_profiles.json', 'w') as outfile:
        json.dump(profiles_with_preferences, outfile, indent=4)

    print("100 customer profiles have been simulated and saved to 'simulated_customer_profiles.json'.")

if __name__ == "__main__":
    main()
