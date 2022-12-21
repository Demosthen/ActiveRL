rm python 
ln -s /.env/python python
chmod +x python
alias python=./python
python -m pip install sinergym[extras]
export PATH=/global/home/users/djang/.conda/envs/ActiveRL/bin:$PATH
