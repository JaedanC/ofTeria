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
}

void PlayState::draw(ofxGameEngine* game)
{
	debugPush("State: PlayState");
}
