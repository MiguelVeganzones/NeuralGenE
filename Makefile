CXX = 				g++

DEBUG_CXXFLAGS =  	-fdiagnostics-color=always \
					-fdiagnostics-show-template-tree \
					-fdiagnostics-path-format=inline-events \
					-fdiagnostics-show-caret \
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
					-fconcepts-diagnostics-depth=3 \
					-std=c++23 

RELEASE_CXXFLAGS =  -fdiagnostics-color=always \
					-fdiagnostics-show-template-tree \
					-fdiagnostics-path-format=inline-events \
					-fdiagnostics-show-caret \
					-O3 \
					-Wall \
					-Wextra \
					-Wshadow \
					-Wconversion \
					-pedantic \
					-Werror \
					-mavx \
					-fbounds-check \
					-fconcepts-diagnostics-depth=3 \
					-std=c++23 
					
PLT_CXXFLAGS =  -fdiagnostics-color=always \
					-fdiagnostics-show-template-tree \
					-fdiagnostics-path-format=inline-events \
					-fdiagnostics-show-caret \
					-O3 \
					-Wall \
					-Wextra \
					-Wshadow \
					-Wconversion \
					-pedantic \
					-mavx \
					-fbounds-check \
					-fconcepts-diagnostics-depth=3 \
					-std=c++23 

RELEASE ?= 0
ifeq (${RELEASE}, 1)
    CXXFLAGS = ${RELEASE_CXXFLAGS}
	OUT_DIR = bin/release
else
    CXXFLAGS = ${DEBUG_CXXFLAGS}
	OUT_DIR = bin/debug
endif

UTILITY_DIR = Utility
MCTS_DIR = Monte\ Carlo\ Tree\ Search
C4_DIR = Connect\ Four
SNN_DIR = Static\ Neural\ Net
PLT_DIR = Plotting
NEURAL_MODEL_DIR = Neural\ Model
MINIMAX_SEARCH_DIR = Minimax\ Tree\ Search
EVOLUTION_AGENT_DIR = Evolution\ Agent

UTILITY_INCL =		
GENERAL_INCL =			-I./$(UTILITY_DIR)
MCTS_INCL =				-I./$(C4_DIR) $(GENERAL_INCL)
C4_INCL =				$(GENERAL_INCL)
SNN_INCL =				$(GENERAL_INCL)
NEURAL_MODEL_INCL = 	-I./$(SNN_DIR) $(SNN_INCL) \
						-I./$(C4_DIR) $(C4_INCL) $(GENERAL_INCL)
PLT_INCL =				-I./$(SNN_DIR) $(SNN_INCL) -I./$(C4_DIR) $(C4_INCL) \
						-I/home/miguelveganzones/Libraries/matplotplusplus-1.1.0-Linux/include \
						-I./$(NEURAL_MODEL_DIR) $(NEURAL_MODEL_INCL) $(GENERAL_INCL)
MINIMAX_SEARCH_INCL = 	-I./$(C4_DIR) $(C4_INCL) $(GENERAL_INCL) \
						-I./$(NEURAL_MODEL_DIR) $(NEURAL_MODEL_INCL) \
						-I./$(SNN_DIR) $(SNN_INCL)
EVOLUTION_AGENT_INCL = 	$(GENERAL_INCL) -I./$(NEURAL_MODEL_DIR) $(NEURAL_MODEL_INCL)

PLT_LIB = 				-L/home/miguelveganzones/Libraries/matplotplusplus-1.1.0-Linux/lib -l:libmatplot.a \
						-L/home/miguelveganzones/Libraries/matplotplusplus-1.1.0-Linux/lib/Matplot++ -l:libnodesoup.a

MINIMAX_LIB = 			-ltbb

all: mcts_main c4_main static_nn_main utility_main minimax_main neural_model_main evolution_agent_main

mcts_main: 
	@mkdir -p $(MCTS_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(MCTS_INCL) $(MCTS_DIR)/$@.cpp -o $(MCTS_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"

c4_main:
	@mkdir -p $(C4_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(C4_INCL) $(C4_DIR)/$@.cpp -o $(C4_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"

static_nn_main:
	@mkdir -p $(SNN_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(SNN_INCL) $(SNN_DIR)/$@.cpp -o $(SNN_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"
	
utility_main:
	@mkdir -p $(UTILITY_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(UTILITY_INCL) $(UTILITY_DIR)/$@.cpp -o $(UTILITY_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"

minimax_main:
	@mkdir -p $(MINIMAX_SEARCH_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(MINIMAX_SEARCH_INCL) ${MINIMAX_LIB} $(MINIMAX_SEARCH_DIR)/$@.cpp -o $(MINIMAX_SEARCH_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"
	
neural_model_main:
	@mkdir -p $(NEURAL_MODEL_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(NEURAL_MODEL_INCL) $(NEURAL_MODEL_DIR)/$@.cpp -o $(NEURAL_MODEL_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"
		
plotting_utility_main:
	@mkdir -p $(PLT_DIR)/${OUT_DIR}
	$(CXX) $(PLT_CXXFLAGS) $(PLT_INCL) $(PLT_LIB) -Xlinker --verbose $(PLT_DIR)/$@.cpp -o $(PLT_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"

evolution_agent_main:
	@mkdir -p $(EVOLUTION_AGENT_DIR)/${OUT_DIR}
	$(CXX) $(CXXFLAGS) $(EVOLUTION_AGENT_INCL) $(EVOLUTION_AGENT_DIR)/$@.cpp -o $(EVOLUTION_AGENT_DIR)/${OUT_DIR}/$@
	@echo Built $@ successfully."\n"

#$(MCTS_DIR)/%.o: %.hpp
#	$(CXX) $(CXXFLAGS) $(GENERAL_INCL) $(MCTS_INCL) $< -c $(MCTS_DIR)/bin/$@

done:
	@echo Built $@ successfully."\n"
