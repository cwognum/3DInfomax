for i in 0 1 2
do
     python3 get_pretrained_fingerprints.py --input-smiles /home/cas/Desktop/biochem_new/all_unique_smiles.npy --batch-size 2 --num-layers-to-drop $i
done
