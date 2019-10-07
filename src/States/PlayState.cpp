#include "PlayState.h"
#include "ofMain.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ConsoleState.h"

PlayState PlayState::instance;

void PlayState::setup()
{
	ofxGameEngine::Instance()->PushState(ConsoleState::Instance());
	KeyboardInput::Instance()->registerAlias("shoot", 'q');
}

void PlayState::update(ofxGameEngine* game)
{
	if (queryAliasPressed("jump")) {
		cout << "PlayState is not getting blocked!\n";
	}
	if (queryAliasPressed("toggleConsole")) {
		ofxGameEngine::Instance()->PushState(ConsoleState::Instance());
	}
	if (queryAliasPressed("shoot") && queryDown(OF_KEY_ALT)) {
		cout << "Playstate shooting + Alt\n";
	}
}

void PlayState::draw(ofxGameEngine* game)
{
	debugPush("State: PlayState");
}
