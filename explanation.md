onfidence Score Calculation Explanation
The confidence score in the Legal Brief Matcher is calculated using a sophisticated multi-factor approach that considers several aspects of legal argument matching. Here's how it works:
1. Base Similarity (0.0-0.7)
This is the foundation of the matching score
Calculated using semantic similarity from the legal-domain transformer model
Measures how closely the content of two arguments align in meaning
Example: A base similarity of 0.55 means the arguments are conceptually related by 55%
2. Citation Boost (0.0-0.2)
Provides additional points when arguments share legal citations
More shared citations = higher boost (capped at 0.2)
Each shared citation contributes approximately 0.05 to the score
Example: 3 shared citations would add a 0.15 boost
3. Heading Match (0.0/0.15)
Binary boost that adds 0.15 if argument headings address the same legal topic
Example: "Likelihood of Success" paired with "Plaintiffs Are Not Likely to Succeed" gets a full 0.15 boost
4. Legal Terminology (0.0-0.2)
Points added for shared legal terms, tests, and standards
More sophisticated terminology matching = higher boost
Example: Shared terms like "irreparable harm" or "balance of equities" increase the score
5. Pattern Match (0.0/0.1)
Adds 0.1 when counter-argument patterns are detected
Example: When one argument makes a claim and the other directly refutes it
6. Length Penalty (-0.1-0.0)
Reduces score for significant disparities in argument length
Calculated as: 0.1 * (1 - min_length/max_length)
Ensures fair matching between arguments of similar scope
Example: If one argument is twice as long as the other, a penalty of about 0.05 is applied
7. Precedent Impact (0.0-0.05)
Small boost based on shared precedent strength
Reflects the importance of cited cases
Example: Strong precedents in both arguments add up to 0.05 to the score
Final Score Calculation:
Apply to stanford_hac...
Penalty
(with a maximum ceiling of 1.0)
This multi-factor approach ensures matches are based not just on text similarity but on legal-specific features that matter to attorneys and judges. The detailed breakdown in the "Explanation for Judges" tab provides transparency about why specific matches are scored highly, helping users understand the reasoning behind each match recommendation.