#include "PlayState.h"
#include "ofMain.h"
#include "ofxDebugger/ofxDebugger.h"

PlayState PlayState::instance;

void PlayState::setup()
{
}

void PlayState::update(ofxGameEngine* game)
{
}

void PlayState::draw(ofxGameEngine* game)
{
	debugPush("State: PlayState");
}
