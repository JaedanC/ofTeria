#include "ofxGameState.h"
#include "../src/Keyboard/KeyboardInput.h"

bool ofxGameState::queryAliasPressed(string& alias)
{
	if (registeredAliasBlocks.count(alias) == 0) {
		registerAliasBlock(alias, INPUT_BLOCK);
	}

	if (aliasPass.count(alias) == 0 || aliasPass[alias]) {
		return KeyboardInput::Instance()->queryAliasPressed(alias);
	}

	return false;
}

void ofxGameState::registerAliasBlock(string alias, KeyboardInputBlockingType blockingType)
{
	registeredAliasBlocks[alias] = blockingType;
}
