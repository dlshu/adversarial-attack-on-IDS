# adversarial-attack-on-IDS
Source code for the paper 'Generative Adversarial Attacks Against Intrusion Detection Systems Using Active Learning'.

Requires the network traffic flow data file 'MachineLearningCSV.zip' from the CICIDS 2017 dataset (available for free at https://www.unb.ca/cic/datasets/ids-2017.html).

To generate the adversarial attack using the Gen-AAL algorithm:
1. Run 'data_preprocessing.py' to generate the data files 'NBx.npy' and NBy.npy.
2. Run 'pretrain_vae.py' to train the IDS model.
3. Run 'pretrain_vae.py' to initialize the VAE model.
4. Run 'attack_Gen-AAL.py' to generate adversarial attacks using the Gen-AAL algorithm and see the attack success rate.
5. For performance comparison, run 'attack_DFAL.py' to generate adversarial attacks using the DFAL algorithm.

The Gen-AAL algorithm will be improved in future updates.
