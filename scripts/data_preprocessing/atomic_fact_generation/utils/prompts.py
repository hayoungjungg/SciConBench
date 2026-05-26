decomposition_persona_description = """You are an expert in breaking down complex scientific sentences into atomic facts--short statements that each contain one piece of information. Given the entire scientific paragraph as context, you will be given one of the sentences and you will need to break the sentence down into atomic facts."""

decontextualization_persona_description = """You are an expert in decontextualizing vague references in scientific sentences into more specific entities, ensuring that any statement relations are not out of context and are self-contained. Given a statement and a response, you will need to decontextualize the vague references in the statement."""


decomposition_prompt="""Using the full scientific paragraph as context, please breakdown one of the sentences from the paragraph into independent atomic facts and justify your answer. The atomic facts should be short statements that each contain one piece of information. Follow the guidelines and examples below to breakdown the sentence into independent atomic facts.

***GUIDELINES FOR GENERATING ATOMIC FACTS STARTS HERE:***
- Make sure that each atomic fact is independent. Each atomic fact in the output should contain different pieces of information.
- Make sure the atomic facts can stand on its own as a fact, using the provided paragraph to contextualize each atomic fact.
- Do not create atomic facts that are inferred or indirect from the sentence. Focus directly on the sentence and the information it contains. For example, if the sentence is regarding the quality of evidence surrounding the use of antibiotics for otitis media with effusion (OME), do not create atomic facts like "Antibiotics are used for otitis media with effusion (OME)." which are inferred from the sentence, but rather focus on the quality of evidence surrounding the use of antibiotics for otitis media with effusion (OME).
- Do not create atomic facts specific to the study itself, but rather focus on the main conclusions. Do not include the systematic review or specific studies as entities in the atomic facts. (e.g., do not create atomic facts like "The review found that..." or "The certainty of the evidence in the systematic review...")
- Do not assume that atomic facts inform one another in terms of context. Treat each atomic fact as a separate, independent fact, each of which needs its own context to be understood on its own. For example, sentence discussing data on adverse effects should be broken down in atomic facts specifying what the adverse effects are on or related to, rather than simply mentioning adverse effects broadly.
- Do not generate atomic facts that reflect authorial judgments or methodological decisions made by the systematic review authors, or that require that context to be understood. For example, "The certainty of the evidence on [TOPIC] was downgraded due to imprecision" reflects the authors' decision to downgrade certainty rather than an intrinsic property of the evidence itself. This does not reflect the current state of the evidence on [TOPIC]. Thus, this atomic fact should instead be "Inconsistency reduced the certainty of the evidence on [TOPIC]."
- DO NOT generate atomic facts for non-content elements such as:
  * Conversational markers (e.g., "Of course", "Happy to help", "Thank you")
  * Meta-commentary about the text (e.g., "This question is excellent", "Here is a detailed breakdown", "Here is an overview...")
  * Markdown formatting (e.g., "###", "---", "***")
  * Section headers and titles (e.g., "Proposed Positive Effects", "Current Limitations")
  * Filler characters and separators (e.g., "---", "***")
  * Very short sentences with no substantive content.
  * Any editorial comments or notes, representing meta-commentary about the review (e.g., "Living systematic reviews offer a new approach to review updating...")
- If the sentence contains only non-content elements, return an empty list of atomic facts.
***GUIDELINES FOR GENERATING ATOMIC FACTS ENDS HERE.***

Here are five examples of how to breakdown a sentence into independent atomic facts:

***EXAMPLE 1 STARTS HERE***
*Full Paragraph:* "Breast stimulation appears beneficial, being associated with a lower number of women not in labour after 72 hours and reduced postpartum haemorrhage rates. However, until safety issues have been fully evaluated, it should not be used in high-risk women. Further research is required to evaluate its safety and should seek data on postpartum haemorrhage rates, the number of women not in labour at 72 hours, and maternal satisfaction."
*Sentence:* "Further research is required to evaluate its safety, and should seek data on postpartum haemorrhage rates, number of women not in labour at 72 hours and maternal satisfaction."
*Atomic facts:*
- "Further research is required to evaluate the safety of breast stimulation."
- "Further research should seek data on postpartum haemorrhage rates."
- "Further research should seek data on the number of women not in labour at 72 hours."
- "Further research should seek data on maternal satisfaction."
*Justification*: The sentence was broken down into four atomic facts, each of which is a short, unique statement that contains one piece of information. Rather than simply containing a vague atomic fact such as "Further research is required," the atomic facts provide more specific information about the safety of breast stimulation, using the paragraph as context to contextualize each atomic fact and helping the reader to understand the specific details on what further research is required, rather than simply knowing that further research is required. Subsequently, the atomic facts were broken down into how further research should data on each individual variables (e.g., maternal satisfaction, postpartam haemorrhage rates, number of women not in labour at 72 hours). 
***EXAMPLE 1 ENDS HERE***

***EXAMPLE 2 STARTS HERE***
*Full Paragraph:* "According to current trials in women undergoing ART, there is no evidence to show that metabolomic assessment of embryos before implantation has any meaningful effect on rates of live birth, ongoing pregnancy, miscarriage, multiple pregnancy, ectopic pregnancy or foetal abnormalities. The existing evidence varied from very low to low-quality. Data on other adverse events were sparse, so we could not reach conclusions on these. At the moment, there is no evidence to support or refute the use of this technique for subfertile women undergoing ART. Robust evidence is needed from further RCTs, which study the effects on live birth and miscarriage rates for the metabolomic assessment of embryo viability. Well designed and executed trials are also needed to study the effects on oocyte quality and endometrial receptivity, since none are currently available."
*Sentence:* "At the moment, there is no evidence to support or refute the use of this technique for subfertile women undergoing ART."
*Atomic facts:*
- "There is currently no evidence to support or refutethe use of metabolomic assessment of embryo viability for subfertile women undergoing ART."
*Justification*: The sentence was broken down into single atomic facts. Rather than simply generating vague atomic facts like "The technique refers to metabolomic assessment of embyros before implanation" and "Subfertile women are undergoing assisted reproductive technology (ART)" which serves more as contexts than facts, the main focus of the sentence is that there is no evidence to support or refute the use of this technique for subfertile women undergoing ART, not on what the technique is and who is undergoing it. Thus, the atomic fact is specific to the sentence -- the sentence is already an atomic fact e.g., a single piece of information on how there is no evidence to support or refute the use of this technique for subfertile women undergoing ART. Importantly, rather than vaguely referring to “the technique" as in the provided sentence, the atomic facts are contextualized using information from the paragraph — specifying that the technique in question is the metabolomic assessment of embryo viability. 
***EXAMPLE 2 ENDS HERE***

***EXAMPLE 3 STARTS HERE***
*Full Paragraph:* "The evidence for the use of antibiotics for OME is of low to very low certainty. Although the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the overall impact on hearing is very uncertain. The long-term effects of antibiotics are unclear and few of the studies included in this review reported on potential harms. These important endpoints should be considered when weighing up the potential short- and long-term benefits and harms of antibiotic treatment in a condition with a high spontaneous resolution rate."
*Sentence:* "The evidence for the use of antibiotics for OME is of low to very low certainty."
*Atomic facts:*
- "The evidence for the use of antibiotics for OME is of low to very low certainty."
*Justification*: The sentence is decomposed into a single atomic facts: the low- and very low-certainty in evidence, which is a single piece of information and conveys that there's no high-quality and certain evidence regarding to the use of antibiotics for OME. Note that atomic facts like "Antibiotics are used for otitis media with effusion (OME)." which are inferred from the sentence are not included because they are not specific to the sentence. Any indirect, or inferred atomic facts from the sentence should not be included.
***EXAMPLE 3 ENDS HERE***  

***EXAMPLE 4 STARTS HERE***
*Full Paragraph:* "The evidence for the use of antibiotics for OME is of low to very low certainty. Although the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the overall impact on hearing is very uncertain. The long-term effects of antibiotics are unclear and few of the studies included in this review reported on potential harms. These important endpoints should be considered when weighing up the potential short- and long-term benefits and harms of antibiotic treatment in a condition with a high spontaneous resolution rate."
*Sentence:* "Although the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the overall impact on hearing is very uncertain."
*Atomic facts:*
- "The use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of otitis media with effusion at up to three months."
- "The overall impact of antibiotic use on hearing in otitis media with effusion is very uncertain."
*Justification*: The sentence was broken down into two atomic facts directly extracted from the sentence. The first atomic fact is specific to the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the second atomic fact is specific to the overall impact of antibiotic use on hearing in OME is very uncertain. Note that these atomic facts are not inferred or indirect from the sentence, but rather directly extracted from the sentence and contextualized using information from the paragraph to be self-contained. 
***EXAMPLE 4 ENDS HERE***  

***EXAMPLE 5 STARTS HERE***
*Full Paragraph:* "At the moment, no evidence from individual RCTs in children and adults with different malignancies underscores use of an LBD for prevention of infection and related outcomes. All studies differed with regard to co-interventions, outcome definitions and intervention and control diets. As pooling of results was not possible, and as all studies had serious methodological limitations, we could reach no definitive conclusions. It should be noted that 'no evidence of effect', as identified in this review, is not the same as 'evidence of no effect'. On the basis of currently available evidence, we are not able to provide recommendations for clinical practice. Additional high-quality research is needed."
*Sentence:* "All studies differed with regard to co-interventions, outcome definitions and intervention and control diets."
*Atomic facts:*
- "Current evidence on the use of low-bacterial diet in preventing infection and related outcomes in children and adults with different malignancies differed with regard to co-interventions across studies."
- "Current evidence on the use of low-bacterial diet in preventing infection and related outcomes in children and adults with different malignancies differed with regard to outcome definitions across studies."
- "Current evidence on the use of low-bacterial diet in preventing infection and related outcomes in children and adults with different malignancies differed with regard to intervention and control diets across studies."
*Justification*: The sentence was broken down into three atomic facts, each of which is a short, unique statement that contains one piece of information. These atomic facts explicitly do not contain out-of-context atomic facts like "All studies differed with regard to co-interventions," which references the "studies" included in the systematic review and requires additional context to be understood e.g., is not self-contained. Thus, the atomic facts are contextualized as "Current evidence on the use of low-bacterial diet in preventing infection and related outcomes in children and adults with different malignancies differed..." and broken down into the specific differences in co-interventions, outcome definitions, and intervention and control diets across studies.
***EXAMPLE 5 ENDS HERE***  

Now, given what you learned from the guidelines and examples, please breakdown the sentence into independent atomic facts, using the provided paragraph and question as context and providing justification and thinking step-by-step about your answer. Avoid creating atomic facts specific to the study itself, but rather focus on the main conclusions.

***QUESTION STARTS HERE*** Use this question as context for your task. For example, if the paragraph is just "We cannot draw any conclusions.", the question should provide context to create atomic facts on how there is not enough evidence to draw any conclusions on the [QUESTION TOPIC]. 
{question}
***QUESTION ENDS HERE***

***FULL PARAGRAPH STARTS HERE*** Use this paragraph as context for your task.
{paragraph}
***FULL PARAGRAPH ENDS HERE***

***SENTENCE STARTS HERE*** 
{sentence} 
***SENTENCE ENDS HERE***

The atomic facts should be short statements that each contain one piece of information. Output should be in JSON format with the following keys:
- "atomic_facts": list of atomic facts as strings
- "justification": string justification for your atomic fact breakdown.
"""

# Edited from https://arxiv.org/pdf/2403.18802
decontextualization_prompt = """Evaluate the provided statement in context of the response. The statement should be decontextualized such that they are understandable without referencing the rest of the response

Instructions:
1. The following STATEMENT has been extracted from the broader context of the given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. For example, given the response “Titanic is a movie starring Leonardo
DiCaprio,” the statement “Titanic is a movie” should not be changed.
5. You MUST NOT generate atomic facts that reflect authorial judgements or methodological decisions made by the systematic review authors.  For example, "The certainty of the evidence on [TOPIC] was downgraded due to imprecision" reflects the authors' decision to downgrade certainty rather than an intrinsic property of the evidence itself. This does not reflect the current state of the evidence on [TOPIC]. Thus, this atomic fact should instead be "Inconsistency reduced the certainty of the evidence on [TOPIC]."
6. Before giving your revised statement, think step-by-step and show your reasoning. As part of your reasoning, be sure to identify the subjects in the
STATEMENT and determine whether they are vague references. If they are vague references, identify the proper entity that they are referring to and be sure
to revise this subject in the revised statement.
7. Your task is to do this for the STATEMENT and RESPONSE under “Your Task”. Some examples have been provided for you to learn how to do this task.

Vague references include but are not limited to:
- Pronouns (e.g., “his”, “they”, “her”)
- Unknown entities (e.g., “this event”, “the research”, “the invention”, "this review", "the study", "many studies")
- Non-full names (e.g., “Jeff...” or “Bezos...” when referring to Jeff Bezos)
- Systematic review, specific studies, or specific evidence (e.g., "the review", "the study", "many studies", "A review", "Our review", "this evidence", "in the review", "in this evidence")
You SHOULD NOT generate atomic facts that reference the systematic review or specific studies in any shape or form e.g., "The certainy of evidence in this systematic review..."

Example 1:
STATEMENT:
Acorns is a company.
RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, Jeff Cruttenden, and Mark Dru that provides micro-investing services. The
company is headquartered in Irvine, California.
REVISED STATEMENT:
The subject in the statement “Acorns is a company” is “Acorns”. “Acorns” is not a pronoun and does not reference an unknown entity. Furthermore, “Acorns”
is not further specified in the RESPONSE, so we can assume that it is a full name. Therefore “Acorns” is not a vague reference. Thus, the revised statement is:
```
Acorns is a company.
```

Example 2:
STATEMENT:
He teaches courses on deep learning.
RESPONSE:
After completing his Ph.D., Quoc Le joined Google Brain, where he has been working on a variety of deep learning projects. Le is also an adjunct professor
at the University of Montreal, where he teaches courses on deep learning.
REVISED STATEMENT:
The subject in the statement “He teaches course on deep learning” is “he”. From the RESPONSE, we can see that this statement comes from the sentence “Le
is also an adjunct professor at the University of Montreal, where he teaches courses on deep learning.”, meaning that “he” refers to “Le”. From the RESPONSE,
we can also see that “Le” refers to “Quoc Le”. Therefore “Le” is a non-full name that should be replaced by “Quoc Le.” Thus, the revised response is:
```
Quoc Le teaches courses on deep learning.
```

Example 3:
STATEMENT:
The television series is called “You’re the Worst.”
RESPONSE:
Xochitl Gomez began her acting career in theater productions, and she made her television debut in 2016 with a guest appearance on the Disney Channel series
“Raven’s Home.” She has also appeared in the television series “You’re the Worst” and “Gentefied.”
REVISED STATEMENT:
The subject of the statement “The television series is called “You’re the Worst.”” is “the television series”. This is a reference to an unknown entity, since it is
unclear what television series is “the television series”. From the RESPONSE, we can see that the STATEMENT is referring to the television series that Xochitl
Gomez appeared in. Thus, “the television series” is a vague reference that should be replaced by “the television series that Xochitl Gomez appeared in”. Thus,
the revised response is:
```
The television series that Xochitl Gomez appeared in is called “You’re the Worst.”
```

Your Task:
QUESTION:
{question}
STATEMENT:
[INDIVIDUAL_FACT]
RESPONSE:
[RESPONSE]

The atomic fact should be short statements that each contain one piece of information. Use the QUESTION as additional context to help understand what the STATEMENT and RESPONSE are referring to. Output should be in JSON format with the following keys:
- "decontextualized_fact": a string of the atomic fact decontextualized to be self-contained.
- "justification": string justification for your decontextualization process of the atomic fact.
"""

####### PROCESSING INCOMPLETE FACTS ####
# Based on VeriFact https://arxiv.org/pdf/2505.09701

detect_incomplete_facts_prompt = """Given a context and a claim extracted from the context, determine whether the claim is Dependent or Independent of the context.
* Independent: If the claim itself precisely reflects the original meaning of the context without further explanation. 
* Dependent: If the claim requires additional context or detail to precisely reflect its original meaning.

Categorize independent claims into one of three types:

* Ambiguous Concepts/Pronouns
    The claim contains vague terms (e.g., "this method," "they") or pronouns lacking clear referents from the context.
    Example:
    Context: "Decarbonizing aviation requires SAFs."
    Claim: "They reduce emissions." → Dependent (Ambiguous pronoun "they").
* Missing Comparison
    The claim implies a comparison (e.g., "more," "better") but omits the explicit comparison target stated in the context.
    Example:
    Context: "SAFs reduce emissions by 80% compared to jet fuels."
    Claim: "SAFs reduce emissions by 80%." → Dependent (Missing "compared to jet fuels").
* Lack of Condition
    The claim omits critical contextual details, such as:
    - Temporal conditions (e.g., "as of 2023").
    - Hypothetical scenarios (e.g., "if regulations are adopted").
    Example:
    Context: "As of 2023, the U.S. top 1% net worth is ~$10M (Smith et al., 2023)."
    Claim: "The U.S. top 1% net worth is ~$10M." → Dependent (Missing time).

ONE CAVEAT: Do not mark claim as Dependent based on not having enough context regarding the scope of claim. The atomic fact is intentionally designed to be self-contained and avoid having any specific details regarding the review or specific studies. Never add rewrites or mark as Dependent based on not having enough context regarding the scope of claim (e.g., "treatments in this review"). Mark such claims as Independent and do not rewrite them.

If the claim is Dependent, you must also provide a rewritten version that makes it Independent by incorporating the missing context. Some guidelines for rewriting dependent claims:
- You MUST NOT not rewrite atomic facts that reflect authorial judgements or methodological decisions made by the systematic review authors. For example, "The certainty of the evidence on [TOPIC] was downgraded due to imprecision" reflects the authors' decision to downgrade certainty rather than an intrinsic property of the evidence itself. This does not reflect the current state of the evidence on [TOPIC]. Thus, this atomic fact should instead be "Inconsistency reduced the certainty of the evidence on [TOPIC]."

Output should be in JSON format with the following keys:
- "classification": string, either "Independent" or "Dependent"
- "dependent_type": string, one of "Ambiguous Concepts/Pronouns", "Missing Comparison", "Lack of Condition", or "None" (if Independent)
- "explanation": string, explanation for your classification
- "rewritten_claim": string, if classification is "Dependent", provide a rewritten version that makes the claim Independent by incorporating missing context. If "Independent", this should be empty. When rewriting this claim, do not include the systematic review or specific studies as entities in the rewritten claim. (e.g., do not rewrite the claim to "The review found that..."). 

# Example

Context:
"Fine-tuning in the context of deep learning refers to the process of taking a pre-trained model—typically one that has been trained on a large dataset—and making small adjustments to its weights and parameters to adapt it for a specific task or dataset. This approach leverages the knowledge the model has already acquired, allowing it to achieve better performance on the new task with less data and training time compared to training a model from scratch. Fine-tuning usually involves freezing some of the earlier layers of the model, which capture general features, while allowing the later layers to be retrained to capture task-specific features. "

Claim:
"Training a model being trained from scratch requires more data."

Your Response:
{
  "classification": "Dependent",
  "dependent_type": "Missing Comparison",
  "explanation": "The claim states that training a model from scratch requires more data, but it does not specify the comparison target for 'more than'.",
  "rewritten_claim": "Training a model from scratch requires more data and training time compared to fine-tuning a pre-trained model."
}

Context:
"The best way to travel from Oakland to Boston depends on your budget, time constraints, and personal preferences. One option is to fly from Oakland International Airport (OAK) to Boston Logan International Airport (BOS) with a major airline such as American Airlines, Delta Air Lines, or United Airlines. Another option is to take a train or bus, but this would be a longer journey, taking around 72 hours with multiple changes. Taking a train would involve boarding the Amtrak Coast Starlight from Oakland to Chicago, then transferring to the Lake Shore Limited to Boston, while taking a bus would involve companies like Greyhound or Megabus with multiple transfers."

Claim:
"Another option for traveling from Oakland to Boston is to take a bus."

Your Response:
{
  "classification": "Dependent",
  "dependent_type": "Ambiguous Concepts/Pronouns",
  "explanation": "The claim mentions 'Another option' without specifying the previous options, making it unclear whether taking a bus is an additional option or the only alternative.",
  "rewritten_claim": "In addition to flying, another option for traveling from Oakland to Boston is to take a bus."
}

Context:
"Decarbonizing civil flight companies is a pressing concern as the aviation industry accounts for approximately 2.5% of global greenhouse gas emissions. If we could have Carbon Offset and Reduction Scheme for International Aviation, a more sustainable aviation industry is possible."

Claim:
"A more sustainable aviation industry is possible."

Your Response:
{
  "classification": "Dependent",
  "dependent_type": "Lack of Condition",
  "explanation": "This claim doesn't mention the condition to make a more sustainable aviation industry possible, which is explicitly stated along with the claim in the given context: 'If we could have Carbon Offset and Reduction Scheme for International Aviation'.",
  "rewritten_claim": "If we could have Carbon Offset and Reduction Scheme for International Aviation, a more sustainable aviation industry is possible."
}

Context:
"The best way to travel from Oakland to Boston depends on your budget, time constraints, and personal preferences. One option is to fly from Oakland International Airport (OAK) to Boston Logan International Airport (BOS) with a major airline such as American Airlines, Delta Air Lines, or United Airlines. Another option is to take a train or bus, but this would be a longer journey, taking around 72 hours with multiple changes. Taking a train would involve boarding the Amtrak Coast Starlight from Oakland to Chicago, then transferring to the Lake Shore Limited to Boston, while taking a bus would involve companies like Greyhound or Megabus with multiple transfers."

Claim:
"The journey from Oakland to Boston by train or bus takes around 72 hours."

Your Response:
{
  "classification": "Independent",
  "dependent_type": "None",
  "explanation": "The claim states that the journey from Oakland to Boston by train or bus takes around 72 hours, which is a self-contained statement and does not require additional context to be understood.",
  "rewritten_claim": ""
}

# Your Task

Context:
"[CONTEXT]"

Claim:
"[CLAIM]"

Your Response:

"""

##### Filtering Irrelevant & Indirect Facts ####
# Based on VeriFact https://arxiv.org/pdf/2505.09701
# AND SAFE: https://arxiv.org/pdf/2403.18802 -- We use the symbols “foo” and “not foo” in lieu of “relevant” and “irrelevant” in order to force the language model to follow our definition of relevance rather than use its own prior knowledge about what relevance means (Wei et al., 2023c,b).

filter_irrelevant_facts_prompt = """A STATEMENT is considered "Foo" if the STATEMENT is directly relevant or provides beneficial background context to addressing the QUESTION in context of the RESPONSE. 

Instructions:
1. The following STATEMENT has been extracted from the broader context of the given RESPONSE to the given QUESTION.
2. First, consider the provided STATEMENT and QUESTION in context of the RESPONSE.
3. Next, determine whether the STATEMENT is considered "Foo" e.g., is it directly relevant or provides beneficial background context to addressing the QUESTION in context of the RESPONSE.
4. Before showing your answer, think step-by-step and show your specific reasoning.
5. If the STATEMENT is considered "Foo", say "[Foo]" after showing your reasoning. Otherwise show "[Not Foo]" after showing your reasoning.
6. Your task is to do this for the STATEMENT and RESPONSE under "Your Task". Some examples have been provided for you to learn how to do this task.

**General Rule of Thumb**
- If a statement is generic response (e.g., "I cannot help with that," "I am not familiar with that," etc.) or does not provide any information that is directly relevant to addressing the QUESTION in context of the RESPONSE, then the STATEMENT is "[Not Foo]".
- If a statement is more indirect or inferred from the RESPONSE, then the STATEMENT is considered "[Not Foo]". The statement should be self-contained, while being directly relevant or providing helpful background context to addressing the QUESTION in context of the RESPONSE. Err to the side of ["Foo"] if the statement provides background context to the RESPONSE. For example, statement on the need for additional high-quality research on some topic and related outcomes provides helpful background context and addresses the QUESTION in context of the RESPONSE.

Example 1:
QUESTION:
What are the benefits and harms of oral antibiotics for otitis media with effusion in children?
RESPONSE:
The evidence for the use of antibiotics for OME is of low to very low certainty. Although the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the overall impact on hearing is very uncertain. The long-term effects of antibiotics are unclear and few of the studies included in this review reported on potential harms. These important endpoints should be considered when weighing up the potential short- and long-term benefits and harms of antibiotic treatment in a condition with a high spontaneous resolution rate.
STATEMENT:
Antibiotics are used for otitis media with effusion (OME).
SOLUTION:
The STATEMENT is indirectly inferred from the RESPONSE (e.g., "The evidence of the use of antibiotics for OME..."). While the statement could be useful background information, it is indirectly inferred from the response and thus is not directly relevant to addressing the QUESTION in context of the RESPONSE. Thus, the STATEMENT is considered "[Not Foo]". Note that the focus of the QUESTION is on the benefits and harms of oral antibiotics for otitis media with effusion in children, which is not directly addressed by the STATEMENT in context of the RESPONSE. The STATEMENT serves as an indirect fact inferred from the RESPONSE and does not directly address the QUESTION. Note that had the RESPONSE contained information that the antibiotics are used for otitis media with effusion (OME), the STATEMENT would have been considered "[Foo]".

Example 2:
QUESTION:
What are the benefits and harms of oral antibiotics for otitis media with effusion in children?
RESPONSE:
The evidence for the use of antibiotics for OME is of low to very low certainty. Although the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the overall impact on hearing is very uncertain. The long-term effects of antibiotics are unclear and few of the studies included in this review reported on potential harms. These important endpoints should be considered when weighing up the potential short- and long-term benefits and harms of antibiotic treatment in a condition with a high spontaneous resolution rate.
STATEMENT:
The use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months.
SOLUTION:
The STATEMENT is directly relevant to the QUESTION in context of the RESPONSE. Thus, the STATEMENT is considered "[Foo]" because the STATEMENT provides information on how the use of oral antibiotics for OME in children may have a slightly beneficial effect on the resolution of OME at up to three months, directly addressing the QUESTION.

Example 3:
QUESTION:
"What are the effects of interventions such as stretching exercises, training schedule modifications, and the use of protective devices on the prevention of lower limb soft-tissue running injuries?"
RESPONSE:
Of course. Based on a synthesis of current sports medicine and physiotherapy research, here is a detailed breakdown of the effects of these common interventions on preventing lower limb soft-tissue running injuries.\n\nIt's important to note that running injuries are often multifactorial, resulting from a complex interaction between **training load** (the primary factor), biomechanics, and an individual's capacity. No single intervention is a magic bullet, and a combined approach is often most effective.
STATEMENT:
The synthesis provides a detailed breakdown of the effects of common interventions. 
SOLUTION:
The STATEMENT is more of a meta-commentary about the text (e.g., The synthesis provides a detailed breakdown of the effects of common interventions.) and does not provide background context nor information that addresses the QUESTION in context of the RESPONSE. The STATEMENT does not provide any information on the effects of interventions on the prevention of lower limb soft-tissue running injuries. Thus, the STATEMENT is considered "[Not Foo]".

Your Task:
QUESTION:
[PROMPT]
RESPONSE:
[RESPONSE]
STATEMENT:
[INDIVIDUAL FACT]

Output should be in JSON format with the following keys:
- "reasoning": string, step-by-step reasoning for your determination
- "classification": string, either "[Foo]" or "[Not Foo]"
"""

filter_redundant_facts_prompt = """A STATEMENT is considered "Foo" ONLY if it conveys essentially the same meaning as exactly one other statement in the given RESPONSE.

Instructions:
1. You are given a RESPONSE and a list of STATEMENTS extracted from that RESPONSE.
2. For each STATEMENT, determine whether it is considered "Foo" given all the other STATEMENTS in the list.
3. A statement is "Foo" ONLY if:
   - It expresses the exact same meaning as one other statement (even if worded differently).
   - It is the same as one other statement, just paraphrased or slightly reworded.
4. A statement is "Not Foo" if:
   - It provides any unique information, even if partially overlapping with other statements
   - It is a subset or superset that adds context or nuance
   - It combines information from multiple statements in a meaningful way
   - Err to the side of "[Not Foo]" if the statement is not the exact same as one other statement and provides *any* partially new information.
5. Before showing your answer, think step-by-step and show your specific reasoning.
6. If the STATEMENT is redundant with exactly one other statement, say "[Foo]" and include that statement number in redundant_with. Otherwise show "[Not Foo]" with an empty redundant_with array.
7. Your task is to do this for the STATEMENT under "Your Task". Some examples have been provided for you to learn how to do this task.

**General Rule of Thumb**
- If a statement is essentially the same as exactly one other statement (paraphrased), it is "[Foo]".
- If a statement is redundant with multiple statements or provides any unique information, it is "[Not Foo]".
- When in doubt, err on the side of "[Not Foo]" -- only mark as "Foo" if it's clearly the same as exactly one other statement, or is a subset of one of the provided statements. For example, "The evidence on the effects of surgical interventions on corneal re-epithelialization in people with neurotrophic keratopathy is of low or very low certainty." is a subset of "The certainty of the evidence on the effects of medical or surgical interventions on corneal re-epithelialization in people with neurotrophic keratopathy is low or very low." and thus is "[Foo]".

Example 1:
RESPONSE:
The evidence for the use of antibiotics for OME is of low to very low certainty. Although the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the overall impact on hearing is very uncertain. The long-term effects of antibiotics are unclear and few of the studies included in this review reported on potential harms.
STATEMENTS:
1. The evidence for the use of antibiotics for OME is of low to very low certainty.
2. The use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months.
3. The overall impact of antibiotics on hearing is very uncertain.
STATEMENT:
The evidence for antibiotics in OME is of low certainty.
SOLUTION:
The STATEMENT is essentially the same as STATEMENT 1 "The evidence for the use of antibiotics for OME is of low to very low certainty", just worded slightly differently and with slightly less detail. The core meaning is the same - both state that the evidence quality is low. Since it conveys essentially the same meaning as exactly one statement (STATEMENT 1), the STATEMENT is [Foo] and redundant_with: ["1"].

Example 2:
RESPONSE:
The evidence for the use of antibiotics for OME is of low to very low certainty. Although the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the overall impact on hearing is very uncertain. The long-term effects of antibiotics are unclear and few of the studies included in this review reported on potential harms.
STATEMENTS:
1. The evidence for the use of antibiotics for OME is of low to very low certainty.
3. The overall impact of antibiotics on hearing is very uncertain.
4. The long-term effects of antibiotics are unclear.
STATEMENT:
The use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months.
SOLUTION:
The STATEMENT provides unique information about the slight beneficial effects regarding the use of antibiotics that is not covered by any of the other STATEMENTS. While the other statements discuss evidence quality, uncertainty, and unclear long-term effects, none mention the slight beneficial effects regarding the use of antibiotics. Thus, the STATEMENT is [Not Foo] and redundant_with: [].

Example 3:
RESPONSE:
Antibiotics can cause side effects including diarrhea, nausea, and vomiting. Some children may experience allergic reactions. The side effects of antibiotics include gastrointestinal symptoms.
STATEMENTS:
1. Antibiotics can cause side effects including diarrhea, nausea, and vomiting.
2. Some children may experience allergic reactions.
3. The side effects of antibiotics include gastrointestinal symptoms.
STATEMENT:
Some of the symptoms of antibiotics include gastrointestinal.
SOLUTION:
The STATEMENT is similar to STATEMENT 3 (which mentions "gastrointestinal symptoms"). Since the STATEMENT is redundant with STATEMENT 3, it is [Foo] and redundant_with: ["3"].

Your Task:
RESPONSE:
[RESPONSE]
STATEMENTS:
[ALL_FACTS]
STATEMENT:
[INDIVIDUAL FACT]

Output should be in JSON format with the following keys:
- "reasoning": string, step-by-step reasoning for your determination
- "classification": string, either "[Foo]" or "[Not Foo]"
- "redundant_with": array of strings (or empty array), the statement number (e.g., ["1"]) from the STATEMENTS list that is most redundant with the given STATEMENT. If the classification is "[Not Foo]", this should be an empty array []. If "[Foo]", include exactly one statement number that expresses the same meaning.
"""

filter_redundant_facts_batch_prompt = """You are given a RESPONSE and a list of STATEMENTS extracted from that RESPONSE. Your task is to select the most atomic set of facts by removing redundant statements, while keeping as many statements that provide as much information about the RESPONSE as possible.

Instructions:
1. You are given a RESPONSE and a numbered list of STATEMENTS extracted from that RESPONSE.
2. Your goal is to select the most atomic set of facts by:
   - Identifying groups of redundant statements (statements that convey exactly the same meaning)
   - For each redundant group, keeping ONLY the best statement(s). Keep the set that retains as many atomic facts as possible while accurately representing the original RESPONSE.
3. A statement is redundant with another if:
   - It expresses the EXACT SAME meaning as another statement (even if worded differently)
   - It is the same as another statement, just paraphrased or *SLIGHTLY* reworded. This includes a subset of another statement (e.g., "low certainty" is redundant with "low or very low certainty")
   - It does not provide any further information beyond what is already covered by other statements.
4. When choosing which statement to keep from a redundant group:
   - STRONGLY prefer more atomic facts: If a statement combines multiple pieces of information (e.g., "diarrhea, nausea, and vomiting"), prefer keeping individual atomic statements (e.g., separate statements for diarrhea, nausea, vomiting) over the combined statement
   - Prefer more specific statements over general ones: If a statement is too broad and does not accurately represent the original RESPONSE, prefer more specific statements that provide comprehensive information
   - Prefer statements that accurately represent the original RESPONSE: Overly general statements that don't provide specific information or don't accurately represent the RESPONSE should be removed in favor of more specific statements
   - Prefer having more atomic facts that each convey minimal information over combined facts that convey multiple pieces of information
5. A statement should be kept if:
   - It provides unique information not covered by any other statement
   - It is the best version in its redundant group (most atomic, most specific, most accurate to RESPONSE)
   - It provides new information not covered by any other statement, even if it's partly new information
   - It accurately represents the original RESPONSE with specific details
   - When in doubt, err on the side of keeping the fact. Only remove if it's clearly the same as exactly one other statement, is a subset of one of the provided statements, or does not provide any further information beyond what is already covered by other statements
6. A statement should be removed if:
   - It is redundant with another statement and does not provide any further information beyond what is already covered
   - It is overly general and does not accurately represent the original RESPONSE
   - It is a subset of another statement that is more complete and does not provide any additional information (e.g., no implications, context, information, etc).
7. Before showing your answer, think step-by-step about:
   - Which statements are redundant with each other
   - Which statements are more atomic (individual facts vs combined facts)
   - Which statements are more specific and accurately represent the RESPONSE
   - Which statements are overly general and should be removed
   - Which statement(s) are best in each redundant group
   - Which statements are unique and should be kept

THINK CAREFULLY STEP-BY-STEP BEFORE EXCLUDING REDUNDANT STATEMENTS. 
Only exclude a statement if it's extremely similar or exact same compared to another statement, or if it's overly general and does not accurately represent the original RESPONSE. KEEP STATEMENTS THAT PROVIDE ADDITIONAL CONTEXT, IMPLICATIONS, OR INFORMATIONS THAT IS NOT COVERED BY OTHER STATEMENTS. For example, 

**General Rule of Thumb**
- STRONGLY prefer more atomic facts: If a statement combines multiple pieces of information, prefer keeping individual atomic statements over the combined statement
- Prefer specific statements over general ones: Overly general statements that don't accurately represent the original RESPONSE should be removed in favor of more specific statements
- Only remove facts that do not provide any further information beyond what is already covered by other statements
- ALWAYS remove statements that are literally the same as another statement (either verbatim or rearranged).
- When multiple statements seem redundant but each provides specific information about different aspects (e.g., different outcomes), keep all of them as they collectively provide comprehensive information
- When in doubt, err on the side of keeping the fact - only remove if it's clearly redundant or overly general

Example 1:
RESPONSE:
The evidence for the use of antibiotics for OME is of low to very low certainty. Although the use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months, the overall impact on hearing is very uncertain. The long-term effects of antibiotics are unclear and few of the studies included in this review reported on potential harms.
STATEMENTS:
1. The evidence for the use of antibiotics for OME is of low to very low certainty.
2. The evidence for antibiotics in OME is of low certainty.
3. The use of antibiotics compared to no treatment may have a slight beneficial effect on the resolution of OME at up to three months.
4. The overall impact of antibiotics on hearing is very uncertain.
SOLUTION:
STATEMENTS 1 and 2 are redundant - they both convey that the evidence quality is low. STATEMENT 1 is more complete ("low to very low certainty" vs "low certainty"), so we keep STATEMENT 1 and remove STATEMENT 2.
STATEMENTS 3 and 4 are unique and should be kept.
Selected statements: [1, 3, 4]

Example 2:
RESPONSE:
Antibiotics can cause side effects including diarrhea, nausea, and vomiting. Some children may experience allergic reactions. The side effects of antibiotics include gastrointestinal symptoms.
STATEMENTS:
1. Antibiotics can cause side effects including diarrhea, nausea, and vomiting.
2. Some children may experience allergic reactions.
3. The side effects of antibiotics include gastrointestinal symptoms.
4. The side effects of antibiotics can include diarrhea.
5. The side effects of antibiotics can include nausea.
6. The side effects of antibiotics can include vomiting.
7. Some of the symptoms of antibiotics include gastrointestinal.
SOLUTION:
STATEMENT 1 combines information about diarrhea, nausea, and vomiting into one statement. Since we prefer more atomic facts that each convey minimal information, we keep the individual atomic statements (STATEMENTS 4, 5, and 6) instead of the combined statement (STATEMENT 1). STATEMENT 2 is unique about allergic reactions and should be kept. STATEMENT 3 provides information about gastrointestinal symptoms and is clearer than STATEMENT 7, so we keep STATEMENT 3 and remove STATEMENT 7. We remove STATEMENT 1 because it's redundant with the combination of STATEMENTS 4, 5, and 6, and we prefer the more atomic facts. Only facts that did not provide any further information beyond what is already covered by other statements were removed.
Selected statements: [2, 3, 4, 5, 6]

Example 3:
RESPONSE:
Our review found low or very low certainty of evidence regarding the effects of medical or surgical interventions on corneal re‐epithelialization, visual acuity, and corneal sensitivity.
STATEMENTS:
1. The certainty of the evidence on the effects of medical or surgical interventions for neurotrophic keratopathy is low or very low.
2. The certainty of the evidence on the effects of surgical interventions on neurotrophic keratopathy is low or very low.
3. The certainty of the evidence on the effects of medical or surgical interventions on corneal re-epithelialization in people with neurotrophic keratopathy is low or very low.
4. The certainty of the evidence on the effects of medical or surgical interventions on visual acuity in people with neurotrophic keratopathy is low or very low.
5. The certainty of the evidence on the effects of medical or surgical interventions on corneal sensitivity in people with neurotrophic keratopathy is low or very low.
SOLUTION:
STATEMENT 1 is too broad and is not a representative to be kept alone in the context of the original RESPONSE. In context of the original RESPONSE, STATEMENTS 3, 4, and 5 are the most specific and provide the most comprehensive information surrounding RESPONSE. While STATEMENTS 3, 4, and 5 seem redundant and may overlap, they all add specific and helpful information surrounding the certainy of the evidence on the effects of medical and surgical interventions on corneal re-epithelialization, visual acuity, and corneal sensitivity in people with neurotrophic keratopathy. Thus, keeping STATEMENTS 3, 4, and 5 would mean STATEMENTS 1 and 2 do not add any new information and are redundant. Statements 1 and 2 are overly general and do not provide any specific information nor do they accurately represent the original RESPONSE. Thus, we remove STATEMENTS 1 and 2 and keep STATEMENTS 3, 4, and 5.
Selected statements: [3, 4, 5]

Your Task:
RESPONSE:
[RESPONSE]
STATEMENTS:
[ALL_FACTS]

Output should be in JSON format with the following keys:
- "reasoning": string, step-by-step reasoning explaining which statements are redundant, which ones you're keeping, and why
- "selected_statements": array of integers, the statement numbers (1-based) from the STATEMENTS list that should be kept in the final atomic set. This should contain no redundant statements and preserve all unique information.
"""