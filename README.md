This branch includes most of the tests that I attempted in solving this inverted pendulum problem. 
I would say that there are 3 important models to see and I will walk you through running my code for each (I wrote them at different times so the logic changed a bit).
I couldn't get TD3 working so that is the only sub-directory you don't need to check out.

First, you can use environment.yml to set up the conda environment. How I do it on my laptop (Windows):
Open a new base terminal (can be in the InvertedPendulumCompetition base directory). Then run:

conda env create -f environment.yml  

conda activate inverted-pendulum-gpu

My first 'working' model uses a manual baseline controller and an SAC (Soft Actor Critic) agent trained just to learn residual control on the base_yaw joint:

Change to the correct directory: cd 'SAC attempts (working)'
You can change the model in YourControlCode.py on line 67 by changing the path.
It should be set to a decently working model already so you just need to run: python Run_Pendulum.py

Video given called "SAC Attempts video.mp4"
<video controls src="SAC attempts video.mp4" title="Title"></video>

Next 'working' model uses a manual baseline controller and an SAC agent trained to learn residual control on the base_yaw, shoulder_pitch, and elbow joints:
To run this switch to the 'Curriculum Attempt' directory. 
Because I was testing different models a bunch, selecting the model is done by changing the environment variable SAC_AGENT_PATH. 
To run with the current final model run: 
$env:SAC_AGENT_PATH="C:\Users\19784\OneDrive\Documents\GitHub\InvertedPendulumCompetition\Curriculum Attempt\artifacts\sac_residual.pt"; python Run_PendulumEnv.py

Video given called "SAC 3 Joints video.mp4"
<video controls src="SAC 3 Joints video.mp4" title="Title"></video>

Finally, I tried a model that trains residuals on all joints. It ended up working a bit differently (not actually better):
To run, switch to the 'All Joints' directory and run:
$env:SAC_AGENT_PATH="C:\Users\19784\OneDrive\Documents\GitHub\InvertedPendulumCompetition\All Joints\artifacts\sac_residual.pt"; python Run_PendulumEnv.py

Video: "SAC All Joints video.mp4"
<video controls src="SAC All Joints video.mp4" title="Title"></video>