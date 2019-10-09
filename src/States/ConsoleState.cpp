#include "ConsoleState.h"
#include "ofxDebugger/ofxDebugger.h"
#include "../Keyboard/KeyboardInput.h"
#include "Console/ConsoleParser.h"
#include "../addons/pystring/pystring.h"

ConsoleState ConsoleState::instance;

ConsoleState::ConsoleState() : ofxGameState(
	true, // updateTransparent
	true, // drawTransparent
	"ConsoleState"
), consoleParser(this)
{
	// 
	: Remove these three lines for testing
	ofxGameEngine::Instance()->getKeyboardInput()->registerAlias("clearConsole", "l", -96); // See -96 is the CTRL modifier
	submitCommand("Hello world");
	submitCommand("Second command");
	registerAliasBlock("jump", INPUT_PASS);
}

void ConsoleState::keyPressed(int key)
{
	/* Controlling the cursor. Enter, Backspace, Delete, Arrow keys, PGUP and PGDOWN. */

	switch (key) {
	case OF_KEY_RETURN:
		submitCommand(currentCommand);
		break;
	case OF_KEY_BACKSPACE:
		if (currentCommand.size() > 0 && cursor.getCursor() > 0) {
			if (cursor.highlighting()) {
				currentCommand.erase(
					currentCommand.begin() + cursor.left(),
					currentCommand.begin() + cursor.right()
				);
				cursor.set(cursor.left());
			} else {
				currentCommand.erase(currentCommand.begin() + cursor.getCursor() - 1);
				--cursor;
			}
			cursor.reset();
		}
		break;
	case OF_KEY_DEL:
		if (currentCommand.size() > 0 && cursor.getCursor() < (int)currentCommand.size()) {
			if (cursor.highlighting()) {
				currentCommand.erase(
					currentCommand.begin() + cursor.left(),
					currentCommand.begin() + cursor.right()
				);
				cursor.set(cursor.left());
			} else {
				currentCommand.erase(currentCommand.begin() + cursor.getCursor());
			}
			cursor.reset();
		}
		break;
	case OF_KEY_UP:
		commandHistoryMarker = ofClamp(--commandHistoryMarker, 0, commandHistory.size());
		currentCommand = (commandHistoryMarker == commandHistory.size()) ? "" : commandHistory[commandHistoryMarker];
		cursor.set(currentCommand.size());
		cursor.reset();
		break;
	case OF_KEY_DOWN:
		commandHistoryMarker = ofClamp(++commandHistoryMarker, 0, commandHistory.size());
		currentCommand = (commandHistoryMarker == commandHistory.size()) ? "" : commandHistory[commandHistoryMarker];
		cursor.set(currentCommand.size());
		cursor.reset();
		break;
	case OF_KEY_LEFT:
		--cursor;
		if (queryInput(OF_KEY_SHIFT, QUERY_DOWN)) {
			break;
		} else {
			cursor.reset();
		}
		break;
	case OF_KEY_RIGHT:
		++cursor;
		if (queryInput(OF_KEY_SHIFT, QUERY_DOWN)) {
			break;
		}
		else {
			cursor.reset();
		}
		break;
	case OF_KEY_PAGE_UP:
		consoleHistoryMarker = ofClamp(--consoleHistoryMarker, 0, consoleHistory.size());
		break;
	case OF_KEY_PAGE_DOWN:
		consoleHistoryMarker = ofClamp(++consoleHistoryMarker, 0, consoleHistory.size());
		break;
	}

	cursor.clamp(0, currentCommand.size());

	// Write key to current string buffer
	if (key >= ' ' && key <= '~' && key != '`') {
		if (cursor.highlighting()) {
			currentCommand.erase(
				currentCommand.begin() + cursor.left(),
				currentCommand.begin() + cursor.right()
			);
			cursor.set(cursor.left());
		}

		currentCommand.insert(currentCommand.begin() + cursor.getCursor(), key);
		++cursor;
		cursor.reset();
	}

	// TODO: Remove
	cout << key << endl;
}

void ConsoleState::keyReleased(int key) {}

void ConsoleState::setup()
{
	ofxGameEngine::Instance()->getKeyboardInput()->registerKeyCallback(this);
}

void ConsoleState::exit()
{
	ofxGameEngine::Instance()->getKeyboardInput()->deregisterKeyCallback(this);
}

void ConsoleState::update(ofxGameEngine* game)
{
	// TODO: Remove
	if (queryInput("jump")) {
		cout << "New query triggered for jump\n";
	}

	if (queryInput(OF_KEY_CONTROL, QUERY_DOWN) && queryInput("jump")) {
		cout << "Control jump!\n";
	}

	if (queryInput("toggleConsole")) {
		cout << "Popping Console State\n";
		ofxGameEngine::Instance()->PopState();
	}

	if (queryInput("clearConsole") && queryInput(OF_KEY_CONTROL, QUERY_DOWN)) {
		submitCommand("clear");
	}
}

void ConsoleState::draw(ofxGameEngine* game)
{
#ifdef _DEBUG
	debugPush("State: ConsoleState");
	debugPush("HistoryMarker: " + ofToString(commandHistoryMarker));
	debugPush("CurrentCommandCursorMarker: " + ofToString(cursor.getCursor()));
#endif

	screenPos = { (float)ofGetWidth() - width - screenGap, (float)screenGap };


	// Draw the console background
	ofSetColor(ofColor::black);
	ofDrawRectangle(screenPos, width, height);

	// Draw a typing marker
	// (Width : 8pt , Height : 11pt ) each character.
	if (cursor.highlighting()) {
		ofSetColor(100, 100, 100);
	} else {
		ofSetColor(240, 240, 240);
	}
	ofSetLineWidth(1.0f);
	ofDrawRectangle(
		screenPos + ofVec2f{ cursor.left() * 8 + 5.0f, 5.0f },
		cursor.dist() * 8.0f + 1.0f,
		11
	);

	// Draw the console history
	ofSetColor(ofColor::white);
	ofDrawBitmapString(currentCommand, screenPos.x + 5, screenPos.y + 15 * 1);
	int linesDrawn = 0;
	for (unsigned int i = consoleHistoryMarker; i < consoleHistory.size(); i++) {
		ConsoleEntry& entry = consoleHistory[consoleHistory.size() - 1 - i];

		ofSetColor(entry.colour);
		for (string& line : entry.message) {
			ofDrawBitmapString(line, screenPos.x + 5, screenPos.y + 15 * (linesDrawn + 2));

			linesDrawn++;
			if (linesDrawn >= showHistoryLines) {
				goto endloop;
			}
		}
	}
endloop:
	return; // You need this or c++ will complain cause of the goto
}

void ConsoleState::submitCommand(const string& command)
{
	if (command == "") {
		return;
	}

	// Have to copy since the string is deleted sometimes
	string commandCpy = command;

	// Split the command into a string vector
	vector<string> parameters;
	pystring::split(commandCpy, parameters);

	bool result = consoleParser.run(parameters);
	commandHistory.push_back(commandCpy);

	vector<string> commandVector = { commandCpy };
	consoleHistory.emplace_back(commandVector, result ? colorPass : colorCommand);

	cullHistory();
	currentCommand.clear();
	cursor.clamp(0, currentCommand.size());
	consoleHistoryMarker = 0;
	commandHistoryMarker = commandHistory.size();
}

void ConsoleState::addText(const ConsoleEntry& entry)
{
	consoleHistory.push_back(entry);
}

void ConsoleState::addText(vector<string>& strings, ofColor colour)
{
	ConsoleEntry entry = { strings, colour };
	addText(entry);
}

void ConsoleState::addText(const string& entry, ofColor colour)
{
	vector<string> entries = { entry };
	addText(entries, colour);
}

void ConsoleState::clearHistory()
{
	currentCommand.clear();
	commandHistory.clear();
	consoleHistory.clear();
}

void ConsoleState::cullHistory()
{
	for (unsigned int i = commandHistoryMaxSize; i < commandHistory.size(); i++) {
		commandHistory.pop_front();
	}

	for (unsigned int i = consoleHistoryMaxSize; i < consoleHistory.size(); i++) {
		consoleHistory.pop_front();
	}
}

