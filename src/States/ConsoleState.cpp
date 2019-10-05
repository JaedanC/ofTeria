#include "ConsoleState.h"
#include "ofxDebugger/ofxDebugger.h"
#include "../Keyboard/KeyboardInput.h"

ConsoleState ConsoleState::instance;

void ConsoleState::setup()
{
	screenPos = { ofGetWidth() - 300.0f, 15.0f };
	width = 285;
	height = 200;

	history.push_back("Hello world");
	history.push_back("Second command");

	KeyboardInput::Instance()->registerAlias("jump", 'b');
}

void ConsoleState::update(ofxGameEngine* game)
{
	if (KeyboardInput::Instance()->queryAliasPressed("jump")) {
		cout << "Alias for jump pressed!\n";
	}
	if (KeyboardInput::Instance()->queryAliasDown("jump")) {
		cout << "Alias for jump down!\n";
	}
	if (KeyboardInput::Instance()->queryAliasReleased("jump")) {
		cout << "Alias for jump released!\n";
	}
}

void ConsoleState::draw(ofxGameEngine* game)
{
	debugPush("State: ConsoleState");
	ofSetColor(ofColor::black);
	ofDrawRectangle(screenPos, width, height);
	ofSetColor(ofColor::white);

	for (unsigned int i = 0; i < history.size(); i++) {
		string& command = history[i];
		ofDrawBitmapString(command, screenPos.x + 5, screenPos.y + 15 * (i + 1));
	}
}

/*
Clears the history queue
*/
void ConsoleState::clearHistory()
{
	for (unsigned int i = 0; i < history.size(); i++) {
		history.pop_front();
	}
}

/*
Limits the size of the history queue to be 'limit' long
*/
void ConsoleState::cullHistory(unsigned int limit)
{
	for (unsigned int i = history.size(); i < limit; i++) {
		history.pop_front();
	}
}

