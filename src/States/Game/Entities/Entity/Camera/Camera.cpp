#include "Camera.h"
#include "../Player.h"

Camera::Camera(Player* player)
	: player(player->weak_from_this())
{
	cout << "Making Camera\n";
}

inline ofVec2f* Camera::getWorldPos()
{
	return nullptr;
}

inline void Camera::setZoom(float zoom)
{
}

void Camera::pushCameraMatrix()
{
}

void Camera::popCameraMatrix()
{
}
