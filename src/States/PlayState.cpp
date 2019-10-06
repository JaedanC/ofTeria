#include "PlayState.h"
#include "ofMain.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ConsoleState.h"

PlayState PlayState::instance;

void PlayState::setup()
{
	ofxGameEngine::Instance()->PushState(ConsoleState::Instance());
}

void PlayState::update(ofxGameEngine* game)
{
	if (KeyboardInput::Instance()->queryAliasPressed("jump")) {
		cout << "PlayState seeing the raw jump alias\n";
	}
	if (queryAliasPressed("jump")) {
		cout << "PlayState is not getting blocked!\n";
	}
	if (queryAliasPressed("toggleConsole")) {
		ofxGameEngine::Instance()->PushState(ConsoleState::Instance());
	}
}

void PlayState::draw(ofxGameEngine* game)
{
	debugPush("State: PlayState");
}
