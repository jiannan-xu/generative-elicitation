import textwrap

from base_active_learning_agent import BaseActiveLearningAgent
from utils import query_api

QUESTION_TYPES = ["yn", "open"]
IMPLEMENTATION = "Python regex"  #["Python regex", "system"]


class GenerativeQuestionsAgent(BaseActiveLearningAgent):
    """Active learning agent that generates questions to identify the target regex."""

    def __init__(self, target_specification_file, engine, openai_cache_file=None, question_type="open", **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file, **kwargs)
        self.question_type = question_type
        assert self.question_type in QUESTION_TYPES, f"Invalid question type: {self.question_type}. Must be one of {QUESTION_TYPES}."

    def get_hypothesis_prompt(self, task_description, interaction_history, broken_regexes=None):
        hypothesis_prompt = textwrap.dedent('''\
            Your task is to collaboratively help someone design a regex that will {task_description}.

            Help them come up with a hypothesis for the regex that they should try, consistent with the previous questions and answers.

            Previous questions and answers:
            {interaction_history}
            
            Previous invalid attempts (these regexes failed to compile):
            {broken_regexes}

            Generate the hypothesis regex without quotes and nothing else:'''
        ).format(
            task_description=task_description,
            interaction_history=self.format_questions_and_answers(interaction_history),
            broken_regexes='\n'.join(broken_regexes),
        )
        print(hypothesis_prompt)
        return [{"role": "user", "content": hypothesis_prompt}]

    def get_question_prompt(self, task_description, question_type, implementation, interaction_history):
        if question_type == "yn":
            question_type_insert = "yes/no question"
        elif question_type == "open":
            question_type_insert = "open-ended question"
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        question_prompt = textwrap.dedent('''\
            Your task is to {task_description}.
                                          
            You should first ask a series of indirect questions to understand the financial situation of the user. We are interested in a user’s income, credit score, debt, loan, etc. Such information is important in determining whether a user meets the eligibility requirement of each credit card. However, because financial information can be very sensitive, we want to ask these questions in an indirect manner.
            For example, you may ask the stability of the income ("What type of employment do you have (full-time, part-time, self-employed, etc.)?"), housing situation ("How long have you lived at your current address?"), monthly financial commitments ("Do you have any regular financial obligations, like child support or loan repayments?"), spending and saving habits ("Do you regularly save a portion of your income?"), debt management ("Are you comfortable with your current level of debt?"), 
            access to credit ("How often do you apply for new credit or loans?"), and family and dependents ("Do you support any family members financially?"). 

            Previous questions:
            {interaction_history}

            Generate the most informative {question_type_insert} that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the {implementation} than the questions that have already been asked. \
             At the same time however, the question should be bite-sized, and not ask for too much at once. {additional_prompt} Generate the {question_type_insert} and nothing else:'''
            ).format(
                implementation=implementation,
                task_description=task_description,
                additional_prompt=getattr(self, "additional_query_note", ""),
                question_type_insert=question_type_insert,
                interaction_history=self.format_questions_and_answers(interaction_history)
            )
        print(question_prompt)
        print("===")
        return [{"role": "user", "content": question_prompt}]
    
    def get_query_prompt(self):
        return self.get_question_prompt(self.task_description, self.question_type, self.implementation, [["[Q]", "[A]"]])

    def generate_active_query(self):
        '''Generates a question for the oracle.'''
        question_prompt = self.get_question_prompt(self.task_description, self.question_type, self.implementation, self.interaction_history)
        question, _ = query_api(question_prompt, self.engine, self.openai_cache, self.openai_cache_file, temperature=self.temperature)
        return question
       
    def generate_oracle_response(self, question):
        '''Generates an oracle response for the question'''
        answer = self.query_oracle_api(question, self.question_type)
        self.interaction_history.append((question, answer))
        return answer
    
    def query_type(self):
        return f"question_{self.question_type}"
