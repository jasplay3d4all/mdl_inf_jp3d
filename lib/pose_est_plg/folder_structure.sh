mkdir human_model_files
mkdir human_model_files/smpl
mkdir human_model_files/smplx
mkdir human_model_files/flame
mkdir human_model_files/mano

unzip SMPL_python_v.1.0.0.zip
cp smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl ./human_model_files/smpl/SMPL_FEMALE.pkl
cp smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl ./human_model_files/smpl/SMPL_MALE.pkl

unzip models_smplx_v1_1.zip
cp mv ./models/smplx/* ./human_model_files/smplx/

unzip smplx_mano_flame_correspondences.zip 
mv MANO_SMPLX_vertex_ids.pkl SMPL-X__FLAME_vertex_ids.npy ./human_model_files/smplx/

unzip expose_data.zip
mv ./data/SMPLX_to_J14.pkl ./human_model_files/smplx/

unzip mano_v1_2.zip
mv ./mano_v1_2/models/MANO_*.pkl ./human_model_files/mano/

unzip FLAME2019.zip 
mv generic_model.pkl ./human_model_files/FLAME_NEUTRAL.pkl
mv flame_* ./human_model_files/flame/

rm -rf ./data/ mano_v1_2/ models/ smpl/ smplx/ __MACOSX/ male_model.pkl female_model.pkl Readme.pdf


pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
bash install.sh
ImportError: cannot import name 'bool' from 'numpy' (/usr/local/lib/python3.10/dist-packages/numpy/__init__.py)
numpy==1.23.1

apt update && apt upgrade
add-apt-repository ppa:kisak/kisak-mesa -y
apt update && apt upgrade
apt-get install libosmesa6-dev freeglut3-dev libglfw3-dev libgles2-mesa-dev libosmesa6

git clone https://github.com/mmatl/pyopengl.git
pip install ./pyopengl

./main/config.py
human_model_path = osp.join(root_dir, "../../share_vol/models/osx/", 'human_model_files') 


PYOPENGL_PLATFORM=osmesa python demo.py --gpu 0 --img_path ./input.png --output_folder ./output/ --decoder_setting wo_decoder --pretrained_model_path ../../../share_vol/models/osx/osx_l_wo_decoder.pth.tar 