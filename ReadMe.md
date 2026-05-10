# Generating a Diffusion Model from scratch
## Project Description
Modern generative AI models, such as those used for high-fidelity image synthesis, are deeply rooted in the physics of non-equilibrium statistical mechanics. In this semester- long, collaborative project, we will explore the equivalence between stochastic differential equations (SDEs), Langevin dynamics, and the "score-matching" techniques that power state-of-the-art diffusion models.
Rather than relying entirely on textbooks or traditional lectures, student teams will use Large Language Models (LLMs) as active research collaborators to derive the underlying mathematics, write the simulation code, and map machine learning concepts back to theoretical physics. The ultimate goal is to critically navigate AI assistance: identifying when the LLM produces rigorous physical derivations, and when it "hallucinates" algebraic errors or unphysical numerical instabilities.

Over 14 weeks, teams will progress through four key blocks:

- Langevin Dynamics: Simulating the forward process and entropy production.
- 2D Manifold Learning: Learning the score function of a geometric distribution (e.g., a spiral).
- High-Dimensional Generation Scaling up to a U-Net architecture to generate MNIST digits.
- The Analytical Limit: Exploring exact Bayesian solutions and theoretical frameworks.

## Workflow and Repository Structure
We are working in one-week sprints, with a goal to be set at the start of each week, and a presentation of our results at the end.

Results of our work are to be saved in the respective folders:
- **presentations**: contains latex or pdf presentations
- **notebooks**: contains our experiments, organised in jupyter notebooks
- **knowledge base**: contains markdown summaries of concepts we learned

The following rules apply:
- **descriptive naming**: filenames should unambigiously describe its content
- **readability**: code and markdown files should be entirely readable and understandable for other project members. This means
    - all important context must be given, but short and concisely
    - notation should be consistent across files
    - no random ai slop nobody understands
