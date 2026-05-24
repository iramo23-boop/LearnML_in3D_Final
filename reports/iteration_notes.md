# Iteration notes

## v1 — baseline behavioral cloning
Goal: create a first working model from collected driving data.  
Expected weakness: the bot may imitate normal driving but fail when it gets far from the path.

## v2 — more recovery samples
Goal: add examples where the expert corrects the bot after drifting toward walls or bad angles.  
Expected improvement: fewer crashes and better checkpoint progress.

## v3 — wider network
Goal: increase model capacity so the network can learn more complex steering patterns.  
Expected improvement: smoother steering and better lap consistency.

## v4 — multi-seed testing
Goal: test whether the model generalizes beyond seed 42.  
Expected improvement: better performance on unseen tracks.

## v5 — action smoothing
Goal: reduce sudden steering changes during inference.  
Expected improvement: fewer oscillations and more stable driving.
