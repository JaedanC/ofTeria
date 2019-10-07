#include "ConsoleParser.h"
#include "Commands.h"
#include "../ConsoleState.h"

ConsoleParser::ConsoleParser(ConsoleState* consoleState)
	: consoleState(consoleState)
{
	// Add bind command
	commandMap["bind"] = {
		"bind",
		"bind <key> <alias> binds a key to an alias.\n",
		2,
		bind
	};
}

void ConsoleParser::run(vector<string>& command)
{
	if (command.size() < 1) {
		return;
	}
	
	if (commandMap.count(command[0]) == 0) {
		// TODO: Make sure this outputs to the console not cout
		consoleState->addText("No such command '" + ofToString(command[0]) + "' exists.\n");
		//cout << "No such command '" << command[0] << "' exists.\n";
		return;
	}

	Command& c = commandMap[command[0]];
	if (c.parameters > (int)command.size() - 1 || c.parameters == -1) {
		// TODO: Make sure this outputs to the console not cout
		consoleState->addText(ofToString(c.command) + ": Incorrect parameter amount. Expected " + ofToString(c.parameters) + ". Provided " + ofToString(command.size() - 1) + "\n");
		consoleState->addText("Usage: " + ofToString(c.commandHelp));
		//cout << c.command << ": Incorrect parameter amount. Expected " << c.parameters << ". Provided " << command.size() - 1 << endl;
		//cout << "Usage: " << c.commandHelp;
		return;
	}
	command.erase(command.begin());
	c.function(command);
}
