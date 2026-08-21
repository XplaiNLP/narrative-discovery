CLAIM_EXTRACTION = """\
You are tasked with analyzing a text and extracting the distinct claims, \
arguments, or narratives it contains.

### GUIDELINES
- Extract each distinct claim, argument, or narrative assertion as a \
separate item.
- Each extracted claim should be concise (1-2 sentences) and \
self-contained — understandable without the original text.
- Write each claim in English. Translate if the source text is in another \
language.
- Normalize each claim: remove text-specific surface details (dates, names \
of minor figures, quote fragments) and express the underlying argument. \
Keep major actors/entities.
- If the text contains only one claim or is very short, return a list \
with that single claim.
- Do not add interpretation or commentary.

### Text
'{text}'
"""

PERSONA_CLAIM_EXTRACTION = """\
Read the following text and extract the 1-3 most important core messages \
as normalized claims.

Rules:
- Each claim must be a standalone, directional assertion (argues FOR or AGAINST something).
- Each claim should be 1-2 sentences, written in English.
- Normalize: remove text-specific details (dates, names of minor figures) and \
  express the underlying argument. Keep major actors/entities.
- Do NOT produce neutral topic labels. Produce specific argumentative claims.
- If the text has only one clear message, return just 1 claim.
- Maximum 3 claims.

Text:
{text}
"""

TEXT_DECOMPOSITION = """\
Goal: Given a text, segment it into multiple semantic units, each containing \
detailed descriptions of specific events or activities.
Perform the following tasks:
1. Provide a summary for each semantic unit while retaining all crucial \
details relevant to the original context.
2. Extract all entities directly from the original text of each semantic \
unit, not from the paraphrased summary. Format each entity name in UPPERCASE. \
You should extract all entities including times, locations, people, \
organizations and all kinds of entities.
3. From the entities extracted in Step 2, list all relationships within the \
semantic unit and the corresponding original context in the form of string \
seperated by comma : "ENTITY_A, RELATION_TYPE, ENTITY_B". The RELATION_TYPE \
could be a descriptive sentence, while the entities involved in the \
relationship must come from the entity names extracted in Step 2. Please make \
sure the string contains three elements representing two entities and the \
relationship type.

requirements:
1. Temporal Entities: Represent time entities based on the available details \
without filling in missing parts. Use specific formats based on what parts \
of the date or time are mentioned in the text.

Each semantic unit should be represented as a dictionary containing three \
keys: semantic_unit (a paraphrased summary of each semantic unit), entities \
(a list of entities extracted directly from the original text of each \
semantic unit, formatted in UPPERCASE), and relationships (a list of \
extracted relationship strings that contain three elements, where the \
relationship type is a descriptive sentence). All these dictionaries should \
be stored in a list to facilitate management and access.


Example:

Text:  In September 2024, Dr. Emily Roberts traveled to Paris to attend the \
International Conference on Renewable Energy. During her visit, she explored \
partnerships with several European companies and presented her latest research \
on solar panel efficiency improvements. Meanwhile, on the other side of the \
world, her colleague, Dr. John Miller, was conducting fieldwork in the Amazon \
Rainforest. He documented several new species and observed the effects of \
deforestation on the local wildlife. Both scholars' work is essential in their \
respective fields and contributes significantly to environmental conservation \
efforts.
Output:
[
  {{
    "semantic_unit": "In September 2024, Dr. Emily Roberts attended the \
International Conference on Renewable Energy in Paris, where she presented \
her research on solar panel efficiency improvements and explored partnerships \
with European companies.",
    "entities": ["DR. EMILY ROBERTS", "2024-09", "PARIS", \
"INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY", "EUROPEAN COMPANIES", \
"SOLAR PANEL EFFICIENCY"],
    "relationships": [
      "DR. EMILY ROBERTS, attended, INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY",
      "DR. EMILY ROBERTS, explored partnerships with, EUROPEAN COMPANIES",
      "DR. EMILY ROBERTS, presented research on, SOLAR PANEL EFFICIENCY"
    ]
  }},
  {{
    "semantic_unit": "Dr. John Miller conducted fieldwork in the Amazon \
Rainforest, documenting several new species and observing the effects of \
deforestation on local wildlife.",
    "entities": ["DR. JOHN MILLER", "AMAZON RAINFOREST", "NEW SPECIES", \
"DEFORESTATION", "LOCAL WILDLIFE"],
    "relationships": [
      "DR. JOHN MILLER, conducted fieldwork in, AMAZON RAINFOREST",
      "DR. JOHN MILLER, documented, NEW SPECIES",
      "DR. JOHN MILLER, observed the effects of, DEFORESTATION on LOCAL WILDLIFE"
    ]
  }},
  {{
    "semantic_unit": "The work of both Dr. Emily Roberts and Dr. John Miller \
is crucial in their respective fields and contributes significantly to \
environmental conservation efforts.",
    "entities": ["DR. EMILY ROBERTS", "DR. JOHN MILLER", \
"ENVIRONMENTAL CONSERVATION"],
    "relationships": [
      "DR. EMILY ROBERTS, contributes to, ENVIRONMENTAL CONSERVATION",
      "DR. JOHN MILLER, contributes to, ENVIRONMENTAL CONSERVATION"
    ]
  }}
]


#########
Real_Data:
#########
Text:{text}

"""

RELATIONSHIP_RECONSTRUCTION = """\
You will be given a string containing tuples representing relationships \
between entities. The format of these relationships is incorrect and needs \
to be reconstructed. The correct format should be: \
'ENTITY_A,RELATION_TYPE,ENTITY_B', where each tuple contains three elements: \
two entities and a relationship type. Your task is to reconstruct each \
relationship in the following format: \
{{'source': 'ENTITY_A', 'relation': 'RELATION_TYPE', 'target': 'ENTITY_B'}}. \
Please ensure the output follows this structure, accurately mapping the \
entities and relationships provided.
Incorrect relationships tuple string:{relationship}
"""

ATTRIBUTE_GENERATION = """\
Generate a concise summary of the given entity, capturing its essential \
attributes and important relevant relationships. The summary should read \
like a character sketch in a novel or a product description, providing an \
engaging yet precise overview. Ensure the output only includes the summary \
of the entity without any additional explanations or metadata. The length \
must not exceed 2000 words but can be shorter if the input material is \
limited. Focus on distilling the most important insights with a smooth \
narrative flow, highlighting the entity's core traits and meaningful \
connections.
Entity: {entity}
Related Semantic Units: {semantic_units}
Related Relationships: {relationships}
"""

DINAM_NARRATIVE = """\
Analyze a list of false information and provide a simple, short narrative \
underlying false intention of all the sentences.

### FALSE INFORMATION:
'{claims}'

### GUIDELINES
- Provide one narrative that best fits all of those false information.
- It must be straightforward, standalone and enough descriptive, so it is \
clear without additional context.
- It must be simple and concise, not longer than 15 words.
- It must be clear enough, to easily be understood by a person who is not \
familiar with the topic.
- It must reflect the false perspective those information underlie.
- It must not reveal it is false narrative.
"""

SUPERCLAIM = """\
Below is a set of claims from the same thematic cluster. Your task is to \
synthesize them into 1-3 super-claims that capture the cluster's core message.

A super-claim is a higher-level claim that groups together multiple specific \
claims. It is a normalized, declarative assertion — a statement asserting that \
something is the case, which one could agree or disagree with.

Important: a super-claim is NOT a newspaper headline, NOT a neutral topic \
label, and NOT a verbose summary. It is a concise, argumentative statement \
that captures what is being argued.

For each super-claim, provide:
- "title": The super-claim itself, formulated as a single normalized \
declarative assertion (one sentence, 10-20 words). It must read as a claim, \
not as a headline or topic.
- "description": A slightly fuller restatement of the same claim with \
additional context (1-2 sentences). Also formulated as an assertion, not as \
a summary of what texts argue.

Do not simply list all claims. Synthesize them into higher-level assertions. \
If multiple claims argue the same point from different angles, combine them \
into one super-claim.

Claims in this cluster:
{content}
"""

PERSONAS = {
    "journalist": """\
You are an experienced investigative journalist. You focus on accountability, \
power dynamics, who benefits, and hidden agendas. You are skeptical of \
official narratives and look for what is NOT being said. When analyzing a \
text, you identify the core claims being pushed — who is blamed, who is \
defended, what action is being called for or resisted.""",
    "political_scientist": """\
You are an academic political scientist specializing in information warfare \
and narrative analysis. You identify geopolitical framing, ideological \
positioning, and rhetorical strategies. When analyzing a text, you extract \
the core political claims — what worldview is being promoted, what threat \
narratives are invoked, and what policy positions are implied.""",
    "conspiracy_analyst": """\
You are a researcher who studies conspiracy theories, disinformation, and \
fringe narratives. You recognize patterns like scapegoating, distrust of \
institutions, hidden enemy narratives, and apocalyptic framing. When analyzing \
a text, you identify the core conspiratorial or counter-institutional claims \
— what is being distrusted, who is the alleged villain, and what alternative \
explanation is being offered.""",
    "fact_checker": """\
You are a professional fact-checker. You focus on extracting the specific \
factual claims and causal assertions made in a text — the checkable \
statements. You separate opinions from factual claims. When analyzing a \
text, you extract the concrete assertions that could be verified or \
debunked — specific numbers, causal links, attributions, and predictions.""",
}
