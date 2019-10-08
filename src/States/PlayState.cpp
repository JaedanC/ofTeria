#include "PlayState.h"
#include "ofMain.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ConsoleState.h"
#include "../Keyboard/KeyboardInput.h"

void PlayState::setup()
{
	ofxGameEngine::Instance()->PushState(ConsoleState::Instance());
	ofxGameEngine::Instance()->getKeyboardInput()->registerAlias("shoot", 'q');
}

void PlayState::update(ofxGameEngine* game)
{
	if (queryInput("jump")) {
		cout << "PlayState is not getting blocked!\n";
	}
	if (queryInput("toggleConsole")) {
		ofxGameEngine::Instance()->PushState(ConsoleState::Instance());
	}
	if (queryInput("shoot") && queryInput(OF_KEY_ALT, QUERY_DOWN)) {
		cout << "Playstate shooting + Alt\n";
	}
}

void PlayState::draw(ofxGameEngine* game)
{
	debugPush("State: PlayState");
}
