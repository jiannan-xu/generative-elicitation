from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_transformers import LongContextReorder
import json
import re


class GenerativeRag:
    def __init__(self, openai_api_key, engine, conversation_history,**kwargs):
        self.openai_api_key = openai_api_key
        self.engine = engine
        self.conversation_history = conversation_history
        self.chat_model = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    def generate_credit_card_recommendation(self):
        # Step 2: Prepare your credit card data
        # We will create a vectorized database of credit card details
        
        # read credit card data from a file
        # with open("credit_card_data.json", "r") as file:
        #     credit_cards = json.load(file)
        
        credit_cards = [
            {
                "name": "Chase Freedom Unlimited",
                "features": "1.5% cash back on all purchases, 5% cash back on travel purchased through Chase Ultimate Rewards, 3% cash back on dining and drugstore purchases",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 15 months on purchases and balance transfers"
            },
            {
                "name": "Citi Double Cash Card",
                "features": "2% cash back on all purchases (1% when you buy, 1% when you pay)",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 18 months on balance transfers"
            },
            {
                "name": "Capital One Quicksilver Cash Rewards Credit Card",
                "features": "1.5% cash back on all purchases",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 15 months"
            },
            {
                "name": "Discover it Cash Back",
                "features": "5% cash back on rotating categories, 1% cash back on all other purchases",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 14 months on purchases and balance transfers"
            },
            {
                "name": "Bank of America Customized Cash Rewards Credit Card",
                "features": "3% cash back on a category of your choice, 2% cash back at grocery stores and wholesale clubs, 1% cash back on all other purchases",
                "eligibility": "Good to excellent credit",
                "benefits": "No annual fee, 0% intro APR for 15 months on purchases and balance transfers"
            },
             {
                "name": "Discover Secured Credit Card",
                "features": "Security Deposit: Requires a refundable security deposit, which acts as your credit line. The minimum deposit is $200, and it can go up to $2,500, depending on what you can afford.\
	            Cash Back Rewards: Earn 2% cash back at gas stations and restaurants (up to $1,000 per quarter), and 1% on all other purchases. Discover will also match all the cash back you earn at the end of the first year ￼.\
	            APR: Regular APR is 28.24% for purchases, with a 10.99% introductory APR for balance transfers for six months.\
	                         No Annual Fee: There are no annual fees",
                "eligibility": "Minimum Age: Must be at least 18 years old. U.S. Residency: Requires a U.S. address, a Social Security number, and a U.S. bank account.\
                No Credit History Required: Suitable for individuals with no or poor credit history. Discover reviews your credit report, but approval odds are good for those looking to build or rebuild their credit",
                "benefits": "Credit Building: Helps build credit by reporting to all three major credit bureaus. After seven months of responsible use, Discover will review your account and may upgrade you to an unsecured card, refunding your security deposit.\
	                            Free Credit Monitoring: Provides free monthly FICO scores and credit alerts for suspicious activity, such as Social Security number monitoring ￼.Zero Fraud Liability: You are not liable for unauthorized charges on your card"
            },
            {
            "name": "Capital One Secured Mastercard",
            "features": "Security Deposit: A refundable security deposit is required, with a minimum of $49, $99, or $200 based on your creditworthiness.\
                        APR: Regular APR is 30.49% (Variable).\
                        No Annual Fee: There is no annual fee.",
            "eligibility": "Minimum Age: Must be at least 18 years old.\
                            No Credit History Required: Suitable for people with no or poor credit history.\
                            U.S. Residency: Requires a U.S. bank account and a Social Security number.",
            "benefits": "Credit Building: Reports to all three major credit bureaus to help you build or improve credit.\
                        Credit Line Increase: You may be automatically considered for a higher credit line in as little as six months with responsible use."
            },
            {
             "name": "Citi Secured Mastercard",
            "features": "Security Deposit: Requires a minimum refundable deposit of $200, up to $2,500.\
                APR: Regular APR is 24.74%.\
                No Annual Fee: No annual fee is charged for this card.",
            "eligibility": "Minimum Age: Must be at least 18 years old.\
                    U.S. Residency: Must have a U.S. bank account and provide a Social Security number.\
                    No Credit History Required: Designed for those looking to build or rebuild credit.",
            "benefits": "Credit Building: Reports to all three credit bureaus to help build credit.\
                Access to your FICO score: Cardholders get access to their FICO score through their online account."
            },
            {
            "name": "OpenSky® Secured Visa® Credit Card",
            "features": "Security Deposit: Requires a refundable deposit starting at $200, which acts as your credit limit.\
                        APR: Regular APR is 22.64% Variable.\
                        No Annual Fee: There is a $35 annual fee for this card.",
            "eligibility": "No Credit Check: Does not require a credit check for approval.\
                            Minimum Age: Must be 18 years or older.\
                            Residency: Requires a U.S. address and bank account.",
            "benefits": "Credit Building: Reports to all three major credit bureaus to help establish or rebuild credit.\
                        No Credit Check: Approval is not based on your credit score."
            },
                {
            "name": "First Progress Platinum Prestige Mastercard® Secured Credit Card",
            "features": "Security Deposit: Requires a refundable security deposit starting at $200.\
                        APR: Regular APR is 13.49% Variable, one of the lowest for secured cards.\
                        Annual Fee: $49 annual fee.",
            "eligibility": "Minimum Age: Must be at least 18 years old.\
                            Residency: Requires a U.S. address and a Social Security number.\
                            No Credit History Required: Designed for people with poor or no credit history.",
            "benefits": "Credit Building: Reports to all three credit bureaus.\
                        Low APR: This card offers one of the lowest APRs for a secured card, making it easier to manage balances."
            },
                {
            "name": "Secured Sable ONE Credit Card",
            "features": "Security Deposit: Requires a refundable deposit starting at $10, with no minimum credit score needed.\
                        APR: Variable APR is 16.24%.\
                        No Annual Fee: There is no annual fee for this card.",
            "eligibility": "Minimum Age: Must be at least 18 years old.\
                            Residency: Must have a U.S. address and bank account.\
                            No Credit History Required: Suitable for those with no credit history.",
            "benefits": "Credit Building: Reports to all three major credit bureaus.\
                        Fast Approval: Approval is quick and does not require a credit check."
            },
                {
            "name": "Discover it® Student Secured Card",
            "features": "Security Deposit: Requires a minimum $200 refundable deposit.\
                        Cash Back Rewards: Earn 2% cash back at gas stations and restaurants and 1% on all other purchases.\
                        APR: Regular APR is 28.24% (Variable).\
                        No Annual Fee: There is no annual fee.",
            "eligibility": "Must be a student with a valid Social Security number.\
                            U.S. Residency: Requires a U.S. address and a U.S. bank account.",
            "benefits": "Cash Back Matching: Discover matches all cash back earned at the end of your first year.\
                        Credit Building: Helps build credit by reporting to all three major credit bureaus."
            }
                ]


        # Convert data to text format
        documents = [
            f"Credit Card: {card['name']}\nFeatures: {card['features']}\nEligibility: {card['eligibility']}\nBenefits: {card['benefits']}"
            for card in credit_cards
        ]

        # Step 3: Create embeddings and a vector store (e.g., FAISS)
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vectorstore = FAISS.from_texts(documents, embeddings)

        # Initialize the retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Set top_k to 3


        # Step 4: Set up the RAG pipeline, retrieving top 5 documents from the vector store
        rag_pipeline = RetrievalQA.from_chain_type(
            llm=self.chat_model,
            retriever=retriever,
            return_source_documents=True  # To return the credit card details used for the recommendation
        )

        # Step 5: Define a function to recommend credit cards based on user preferences
        def document_to_string(doc):
            return doc.page_content
        
        def extract_credit_card_names(recommendation):
            # Use regex to extract all credit card names from the recommendation
            card_names = [card['name'] for card in credit_cards]
            pattern = "|".join(re.escape(name) for name in card_names)
            
            # Find all matches for the credit card names in the recommendation
            matches = re.findall(pattern, recommendation)
            
            # Create a set to track the names we've already seen
            seen_names = set()
            unique_names = []
            
            # Iterate over the matches and add only the first occurrence of each name
            for name in matches:
                if name not in seen_names:
                    unique_names.append(name)
                    seen_names.add(name)
            
            return unique_names

        def recommend_credit_cards(user_query):
            result = rag_pipeline({"query": user_query})

            print(result)

            recommended_card_names = extract_credit_card_names(result["result"])

            print(recommended_card_names)

            # Sort the source documents to match the order of the mentioned cards in the result
            sorted_documents = sorted(
                result["source_documents"],
                key=lambda doc: next((i for i, name in enumerate(recommended_card_names) if name in doc.page_content), float('inf'))
            )

            print(sorted_documents)
            
            return {
                "recommendation": result["result"],
                "source_documents": sorted_documents
            }
        
        # Step 6: Get user input and recommend credit cards
        # transform the user query into a string
        user_query = ", ".join(f"{item['sender']}: {item['message']}" for item in self.conversation_history)
        user_query_updated = user_query + "If, based on the conversation, it seems that the user's financial situation is below average or poor, please recommend only the credit cards for which they are likely to meet the eligibility criteria. Recommend only 3 credit cards in total and give the user a brief explanation of why you are recommending these cards."
        recommendation = recommend_credit_cards(user_query_updated)
               
        return recommendation

        # # Display the results
        # print("Recommended Credit Card(s):")
        # print(recommendation["recommendation"])

        # print("\nDetails from Source Documents:")
        # for doc in recommendation["source_documents"]:
        #     print(doc)

