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
	if (queryAliasPressed(string("jump"))) {
		cout << "PlayState is not getting blocked!\n";
	}
}

void PlayState::draw(ofxGameEngine* game)
{
	debugPush("State: PlayState");
}
