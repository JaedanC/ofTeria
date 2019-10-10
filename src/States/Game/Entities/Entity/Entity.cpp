#include "Entity.h"
#include "../EntityController.h"

Entity::Entity(EntityController* entityController)
	: entityController(entityController->weak_from_this())
{
}
