#include "PlayState.h"
#include "ofMain.h"
#include "ofxDebugger/ofxDebugger.h"
#include "ConsoleState.h"
#include "../Keyboard/KeyboardInput.h"

void PlayState::setup()
{
	ofxGameEngine::Instance()->getKeyboardInput()->registerAlias("shoot", "q");
}

void PlayState::update(ofxGameEngine* game)
{
	elapsedTime += ofGetLastFrameTime();
	int passes = floor(elapsedTime * fixedUpdateRate);
	elapsedTime -= passes * (1 / (double)fixedUpdateRate);
	framePercentage = ofMap(elapsedTime, 0, 1.0 / fixedUpdateRate, 0, 1);

	debugPush("Passes: " + ofToString(passes));
	debugPush("fixedUpdateRate: " + ofToString(fixedUpdateRate));
	debugPush("framePercentage: " + ofToString(framePercentage));
	debugPush("elapsedTime: " + ofToString(elapsedTime));
	while (passes--) {
		fixedUpdate(game);
	}

	if (queryInput("toggleConsole")) {
		ofxGameEngine::Instance()->PushState(ConsoleState::Instance());
	}

	getWorldSpawn().lock()->update();
}

void PlayState::fixedUpdate(ofxGameEngine* game)
{
	getWorldSpawn().lock()->fixedUpdate();
}

void PlayState::draw(ofxGameEngine* game)
{
	getWorldSpawn().lock()->draw();
}
