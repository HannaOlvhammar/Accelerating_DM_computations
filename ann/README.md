# Train an artificial neural network on a set of features and labels generated in
# another directory. Two models were trained: One with one input parameter,
# the DM mass, and one with three input parameters consisting of the mass and
# two EFT coefficients.

# One-input model:
Train with mass_nn.py
The model is contained in model_mass_1e5
Analyse further with post_mass_nn.py
Check accuracy and speed with error_speed_mass_nn.py

# Three-input model:
Train with eft_nn.py
The model is contained in model_eft_1e6
Analyse further with post_eft_nn.py
Check accuracy and speed with error_speed_eft_nn.py
