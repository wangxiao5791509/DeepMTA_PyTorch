###################################################################################################
#################################        GOT10K         ###########################################
###################################################################################################
python testing.py -d GOT10k -t SiamRPN --lb_type ensemble


###################################################################################################
#################################        LaSOT          ###########################################
###################################################################################################
python testing.py -d LaSOT -t SiamRPN --lb_type ensemble

python lasot_thor_testing.py -d LaSOT -t SiamRPN --lb_type ensemble


###################################################################################################
#################################        UAV20L          ##########################################
###################################################################################################
python testing.py -d UAV20L -t SiamRPN --lb_type ensemble

###################################################################################################
#################################        UAV123          ##########################################
###################################################################################################
python testing.py -d UAV123 -t SiamRPN --lb_type ensemble

###################################################################################################
#################################        OXUVA           ##########################################
###################################################################################################
python testing.py -d OXUVA -t SiamRPN --lb_type ensemble


