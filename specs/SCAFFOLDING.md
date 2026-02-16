# Specifications for the HyperThink scaffolding

HyperThink is an LLM scaffolding alghorithm that allows two models to run in a continoous feedback loop with a shared auto-decaying state in order to achieve complex reasoning without context or time constraints.

## I. Alghorithm

We define the first model as A and the second model as B.
We set the temperature of A to an higher value (by default, 1.2) and the temperature of B to 0 (or near).
We set the top_p of A to an higher value (by default, 0.95) and the top_p of B to a lower value (by default, 0.2).
Both models must have their reasoning mode enabled.

The alghorithm needs an auto-decaying state, expressed as an array of strings with a given maximum amount of elements (by default, 17).

This is the procedure followed for every query executed:

1.  We run the inference of the model A using the data given by the user (ex. current chat history)
2.  We pass the result to model B, which reviews the input using a structured output which produce either the confirmation of the given input or a new version, together with a list of 2 to 8 notes.
3.  If the input is reviewed as correct, exit the loop and return the output to the user
4.  If the input isn't reviewed as correct, add the given correction notes to the auto-decaying state.
5.  If there was an attempt to a number of notes to the state higher than the amount of avaible slots, remove an amount of random elements equal to the amount of slots that need to be empty.
6.  Re-execute the review loop using model A (from step 2 to step 5)
7.  If the input isn't reviewed as correct, go back to step 2 using model B.

## II. System Prompts

In order to be used, the HyperThink scaffolding needs two system prompts:
- A starter prompt, used for the first inference (step 1 of the alghorithm) and designed to generate a first viable solutions to the query in an unstructured form
- A reviewer prompt, used for the all of the other inferences and designed to review another LLM's output, generating a structured output

## III. Structured Output

All reviewer inferences (so from the second to the last inference) must return a structured outputs, containing:
- `review_result`: Boolean; contains the true/false result of the review (if true, the given input is considered correct)
- `added_notes`: List of strings; contains the notes that the model wants to add to the semi-decaying state
- `output`: String; model's output to the user's query
