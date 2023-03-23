CXX = 				g++

CXXFLAGS =  		-fdiagnostics-color=always \
					-fdiagnostics-show-template-tree \
					-fdiagnostics-path-format=inline-events \
					-fdiagnostics-show-caret \
					-undefined \
					-g \
					-Wall \
					-Wextra \
					-Wshadow \
					-Wconversion \
					-fsanitize=address \
					-fsanitize=leak \
					-fsanitize=undefined \
					-pedantic \
					-Werror \
					-mavx \
					-fbounds-check \
					-std=c++23 

GENERAL_INCL =		-I./Utility 
MTCS_INCL = 		-I./Connect\ Four
C4_INCL = 		-I./Connect\ Four

MTCS_DIR = Monte\ Carlo\ Tree\ Search
C4_DIR = Connect\ Four

mtcs: 
	$(CXX) $(CXXFLAGS) $(GENERAL_INCL) $(MTCS_INCL) $(MTCS_DIR)/mcts_main.cpp -o $(MTCS_DIR)/bin/mcts_main
c4: 
	$(CXX) $(CXXFLAGS) $(GENERAL_INCL) $(C4_INCL) $(C4_DIR)/c4_main.cpp -o $(C4_DIR)/bin/c4_main

#$(MTCS_DIR)/%.o: %.hpp
#	$(CXX) $(CXXFLAGS) $(GENERAL_INCL) $(MTCS_INCL) $< -c $(MTCS_DIR)/bin/$@