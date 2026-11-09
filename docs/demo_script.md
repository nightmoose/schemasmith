# Demo script

About two and a half minutes when read aloud. Numbers stay TBD.

Open the page. The banner says mock mode is on and the scorecard is TBD. That is intentional. The model is not loaded.

Start with the contract on the left. One ticket becomes one JSON object. The fields are the id, the tier, the category, a severity from 1 to 4, an order reference or null, a refund amount or null, and the language code. The rule on the card is the one that matters for money: a refund is allowed only when the category is billing.

Load a sample. The ticket is short and not in English only. Point at the order id and the amount in the text. Those values have to be copied, not invented.

Press Extract. Say clearly that this JSON is the label for the sample, not a generation. Then press Validate only. The contract check is the real validator: it parses the object and runs the schema. If you paste a refund onto a bug ticket, the check fails. If you invent an order id that is not in the text, the unsupported-field line names it.

Move to the two code boxes. The left one is a planted failure from the preference data. The right one is the labeled target. Neither box is a before-and-after from Tiny Aya. The real comparison is a command that runs the base model, the supervised adapter, and the preference adapter on the held-out split and writes a JSON file. The table on this page reads that file and otherwise stays TBD.

Close on the pipeline. CuratorKIT filters the small multilingual set. AlignTune trains a LoRA, then a preference LoRA, on a free GPU. AuditKIT can score beside the contract. The code in the repository is MIT. The Tiny Aya weights are non-commercial. The Lexsi tools stay outside the tree. When the compare file exists, load it here and read only the numbers it contains.

If someone asks for the score, say it has not been measured yet. Point at the TBD row rather than guessing a percent. The held-out file, the seed, and the notebook steps are in the reproducibility guide, so the same split can be scored again later.
