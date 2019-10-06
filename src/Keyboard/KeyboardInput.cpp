#include "KeyboardInput.h"

KeyboardInput KeyboardInput::instance;

void KeyboardInput::keyPressed(int key)
{
	// Update keyPressed Maps
	keyDownMap[key] = true;
	keyPressedMap[key] = true;

	// Run callback functions for the instances that have registered to recieve them also
	for (auto& callback : keyPressedCallbacks) {
		callback->keyPressed(key);
	}

	//TODO: Remove
	cout << key << endl;
}

void KeyboardInput::keyReleased(int key)
{
	// Update keyPressed Maps
	keyDownMap[key] = false;
	keyReleasedMap[key] = true;

	// Run callback functions for the instances that have registered to recieve them also
	for (auto& callback : keyReleasedCallbacks) {
		callback->keyReleased(key);
	}
}

void KeyboardInput::mousePressed(int x, int y, int button)
{
	// TODO
}

void KeyboardInput::mouseReleased(int x, int y, int button)
{
	// TODO
}

void KeyboardInput::registerKeyPressedCallback(KeyboardCallbacks* callbackInstance)
{
	// Only add if it's not already inside the set
	if (keyPressedCallbacks.count(callbackInstance) == 0) {
		keyPressedCallbacks.insert(callbackInstance);
	}
}

void KeyboardInput::registerKeyReleasedCallback(KeyboardCallbacks* callbackInstance)
{
	// Only add if it's not already inside the set
	if (keyReleasedCallbacks.count(callbackInstance) == 0) {
		keyReleasedCallbacks.insert(callbackInstance);
	}
}

void KeyboardInput::deregisterKeyPressedCallback(KeyboardCallbacks* callbackInstance)
{
	// Only remove if it's not already inside the set
	if (keyPressedCallbacks.count(callbackInstance) != 0) {
		keyPressedCallbacks.erase(callbackInstance);
	}
}

void KeyboardInput::deregisterKeyReleasedCallback(KeyboardCallbacks* callbackInstance)
{
	// Only remove if it's not already inside the set
	if (keyReleasedCallbacks.count(callbackInstance) != 0) {
		keyReleasedCallbacks.erase(callbackInstance);
	}
}

void KeyboardInput::registerAlias(string alias, int key)
{
	aliasMappings[alias].push_back(key);
}

bool KeyboardInput::queryAliasPressed(string alias)
{
	// Assume that Alias's that do not exist always return false
	if (aliasMappings.count(alias) == 0) return false;

	// Return true as soon as one of the keys for the binding have been triggered
	for (int& key : aliasMappings[alias]) {
		if (queryPressed(key)) return true;
	}
	return false;
}

bool KeyboardInput::queryAliasReleased(string alias)
{
	// Assume that Alias's that do not exist always return false
	if (aliasMappings.count(alias) == 0) return false;

	// Return true as soon as one of the keys for the binding have been triggered
	for (int& key : aliasMappings[alias]) {
		if (queryReleased(key)) return true;
	}
	return false;
}

bool KeyboardInput::queryAliasDown(string alias)
{
	// Assume that Alias's that do not exist always return false
	if (aliasMappings.count(alias) == 0) return false;

	// Return true as soon as one of the keys for the binding have been triggered
	for (int& key : aliasMappings[alias]) {
		if (queryDown(key)) return true;
	}
	return false;
}

bool KeyboardInput::queryPressed(int key)
{
	/* If the key has not been pressed return false. Otherwise return
	what the value is. */
	return (keyPressedMap.count(key) == 0) ? false : keyPressedMap[key];
}

bool KeyboardInput::queryReleased(int key)
{
	/* If the key has not been pressed return false. Otherwise return
	what the value is. */
	return (keyReleasedMap.count(key) == 0) ? false : keyReleasedMap[key];
}

bool KeyboardInput::queryDown(int key)
{
	/* If the key has not been pressed return false. Otherwise return
	what the value is. */
	return (keyDownMap.count(key) == 0) ? false : keyDownMap[key];
}

void KeyboardInput::resetPollingMaps()
{
	keyPressedMap.clear();
	keyReleasedMap.clear();
}
