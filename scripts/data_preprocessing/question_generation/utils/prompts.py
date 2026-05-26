"""
Prompts for question generation from research objectives.
"""

persona_description = """You are an expert specialized in converting research objectives into their corresponding research question format."""

zero_shot_prompt = """Given the following research objective, please convert it into a single research question, using the provided study background as context to inform your answer. 

***RESEARCH OBJECTIVE STARTS HERE***
{objective}
***RESEARCH OBJECTIVE ENDS HERE***

***STUDY BACKGROUND CONTEXT STARTS HERE***
{background_context}
***STUDY BACKGROUND CONTEXTS ENDS HERE***

Now, using the study background as context, convert this objective into a single research question, justifying your answer and thinking step-by-step about your answer. USE THE GUIDELINES BELOW TO INFORM YOUR OUTPUT QUESTION AND JUSTIFICATION. 

***GUIDELINES FOR GENERATING RESEARCH QUESTIONS STARTS HERE:***
- The generated question should be specific, answerable, and capture the main research focus in the objective. 
- DO NOT INCLUDE ANY EXTRANEOUS INFORMATION OR CONTEXT IN YOUR ANSWER THAT WERE NOT PROVIDED IN THE STUDY BACKGROUND
- DO NOT ADD ADDITIONAL INFORMATION BEYOND WHAT IS STATED IN THE OBJECTIVE. Only use the background as context to inform your answer. 
- Avoid copying the objective verbatim and aim to generate a question that is Google-Proof. Preserve the objective's semantics and the objective to maintain the full population, intervention, comparison, and outcome (PICO) of the study and objective in the generated question.
***GUIDELINES FOR GENERATING RESEARCH QUESTIONS ENDS HERE.***

Output should be in JSON format with the following structure:
- "question": string research question
- "justification": string justification for your answer regarding the converted question.
"""

few_shot_prompt = """Given the following research objective, please convert it into a single question, using the provided study background as context to inform your answer.

***RESEARCH OBJECTIVE STARTS HERE***
{objective}
***RESEARCH OBJECTIVE ENDS HERE***

***STUDY BACKGROUND CONTEXT STARTS HERE***
{background_context}
***STUDY BACKGROUND CONTEXTS ENDS HERE***

Here is an example of how to convert research objectives into single research questions:

***EXAMPLE 1 STARTS HERE***
Research Objective: "To evaluate the effectiveness and safety of metabolomic assessment of oocyte quality, embryo viability, and endometrial receptivity for improving live birth or ongoing pregnancy rates in women undergoing ART, compared to conventional methods of assessment."

Generated Question: "For women receiving assisted reproductive technologies, how effective and safe is metabolomic assessment of oocyte quality, embryo viability, or endometrial receptivity, compared with conventional assessment methods, in improving live birth and ongoing pregnancy rates?"
***EXAMPLE 1 ENDS HERE***

***EXAMPLE 2 STARTS HERE***
Research Objective: "To assess the efficacy and safety of drugs used to ease breathlessness in people with cystic fibrosis."

Generated Question: "How effective and safe are drugs used to alleviate breathlessness in people with cystic fibrosis?"
***EXAMPLE 2 ENDS HERE***

***EXAMPLE 3 STARTS HERE***
Research Objective: "To assess whether: 1. Lithium alone is an effective treatment for schizophrenia, schizophrenia‐like psychoses and schizoaffective psychoses; and 2. Lithium augmentation of antipsychotic medication is an effective treatment for the same illnesses."

Generated Question: "Is lithium--either alone or as an augmentation of antipsychotic medication--an effective treatment for schizophrenia, schizophrenia‐like psychoses and schizoaffective psychoses?"
***EXAMPLE 3 ENDS HERE***

Now, given what you learned from the examples and using the study background as context, please convert the provided objective into a single research question, justifying your answer and thinking step-by-step about your answer. USE THE GUIDELINES BELOW TO INFORM YOUR OUTPUT QUESTION AND JUSTIFICATION. 

***GUIDELINES FOR GENERATING RESEARCH QUESTIONS STARTS HERE:***
- The generated question should be specific, answerable, and capture the main research focus in the objective. 
- DO NOT INCLUDE ANY EXTRANEOUS INFORMATION OR CONTEXT IN YOUR ANSWER THAT WERE NOT PROVIDED IN THE STUDY BACKGROUND
- DO NOT ADD ADDITIONAL INFORMATION BEYOND WHAT IS STATED IN THE OBJECTIVE. Only use the background as context to inform your answer. 
- Avoid copying the objective verbatim and aim to generate a question that is Google-Proof. Preserve the objective's semantics and the objective to maintain the full population, intervention, comparison, and outcome (PICO) of the study and objective in the generated question.
***GUIDELINES FOR GENERATING RESEARCH QUESTIONS ENDS HERE.***

Output should be in JSON format with the following structure:
- "question": string research question
- "justification": string justification for your answer regarding the converted question.
"""
