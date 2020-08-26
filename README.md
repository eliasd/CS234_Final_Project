# Policies for Networking Offloading with Deadlines | CS 234 Final Project
*Authors: Ahmed Ahmed, Daniel Guillen*

## Info
Repo contains code and paper.

## Abstract
Robotic systems are turning toward using computationally-intensive 
resources like DeepNeural Networks (DNN) for different perception, decision making, and localization tasks. 
Additionally, more projects have also started to explore cloud computation for “offloading” DNN queries, 
which is interesting because they can provide improved accuracy and increased compute resources. 
We aim to extend the problem of how resource-constrained systems can optimally choose to offload a task to a cloud-server model over the local 
model by incorporating a “deadline“ that requires a response within a certain amount of time. 
We also encode an instance of the problem where an image-processing system chooses which model to query, while minimizing latency, accounting for 
the deadline, and maximizing prediction accuracy of the given task.
