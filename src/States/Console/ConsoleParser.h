#pragma once
#ifndef CONSOLE_PARSER_H
#define CONSOLE_PARSER_H

#include "ofMain.h"
class ConsoleState;

struct Command {
	string command;
	string commandHelp;
	int parameters; //-1 = inf, 0 = None (god)
	void(*function)(vector<string>);
};

class ConsoleParser {
private:
	ofColor colorFailed = ofColor(255, 100, 100);

	ConsoleState* consoleState;
	unordered_map<string, Command> commandMap;
public:
	explicit ConsoleParser(ConsoleState* consoleState);

	bool run(vector<string> command);
};

#endif /* CONSOLE_PARSER_H */