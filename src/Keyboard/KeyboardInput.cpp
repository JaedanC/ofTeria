#include "KeyboardInput.h"

void KeyboardInput::keyPressed(int key)
{
	// Update keyPressed Maps
	keyDownMap[key] = true;
	keyPressedMap[key] = true;

	// Run callback functions for the instances that have registered to recieve them also
	for (auto& callback : keyPressedCallbacks) {
		callback->keyPressed(key);
	}
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

void KeyboardInput::registerCallback(KeyboardCallbacks* callbackInstance, CallbackType callbackType)
{
	set<KeyboardCallbacks*> * callbackSet;
	switch (callbackType) {
	case CALLBACK_PRESSED:
		callbackSet = &keyPressedCallbacks;
		break;
	case CALLBACK_RELEASED:
		callbackSet = &keyReleasedCallbacks;
		break;
	}

	if (callbackSet->count(callbackInstance) == 0) {
		callbackSet->insert(callbackInstance);
	}
}

void KeyboardInput::deregisterCallback(KeyboardCallbacks* callbackInstance, CallbackType callbackType)
{
	set<KeyboardCallbacks*>* callbackSet;
	switch (callbackType) {
	case CALLBACK_PRESSED:
		callbackSet = &keyPressedCallbacks;
		break;
	case CALLBACK_RELEASED:
		callbackSet = &keyReleasedCallbacks;
		break;
	}

	if (callbackSet->count(callbackInstance) != 0) {
		callbackSet->erase(callbackInstance);
	}
}

void KeyboardInput::registerAlias(string alias, int key)
{
	aliasMappings[alias].push_back(key);
}

bool KeyboardInput::queryInput(const string& alias, QueryType queryType)
{
	// Assume that Alias's that do not exist always return false
	if (aliasMappings.count(alias) == 0) return false;

	// Return true as soon as one of the keys for the binding have been triggered
	for (int& key : aliasMappings[alias]) {
		if (queryInput(key, queryType)) return true;
	}
	return false;
}

bool KeyboardInput::queryInput(const int key, QueryType queryType)
{
	/* If the key has not been pressed return false. Otherwise return
	what the value is. */
	switch (queryType) {
	case QUERY_PRESSED:
		return (keyPressedMap.count(key) == 0) ? false : keyPressedMap[key];
	case QUERY_DOWN:
		return (keyDownMap.count(key) == 0) ? false : keyDownMap[key];
	case QUERY_RELEASED:
		return (keyReleasedMap.count(key) == 0) ? false : keyReleasedMap[key];
	}

	cout << "KeyboardInput::queryInput: queryType not recognised. Returning false\n";
	return false;
}

void KeyboardInput::resetPollingMaps()
{
	keyPressedMap.clear();
	keyReleasedMap.clear();
}
