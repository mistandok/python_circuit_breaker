"""Модуль содержит реализацию circuit_breaker для ручек API."""
from datetime import timedelta
from enum import Enum
from functools import wraps
from time import monotonic
from typing import Callable, Optional


class BreakerState(Enum):
    """Класс перечисляет возможные состояния, в которых может находится CircuitBreaker"""

    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'


class CircuitBreaker:
    """Класс обеспечивает снижение нагрузки на ручки, которые отказали, путем реализации паттерна Circuit breaker."""

    FAILURE_COUNT = 2
    SUCCESS_COUNT = 2
    TIMEOUT = timedelta(seconds=30).total_seconds()
    HANDLE_EXCEPTIONS = (Exception,)
    EXCLUDED_EXCEPTIONS = None
    FALLBACK_FUNCTION = None

    def __init__(
        self,
        failure_count: int | None = None,
        success_count: int | None = None,
        timeout: timedelta | None = None,
        handle_exceptions: tuple | None = None,
        excluded_exceptions: tuple | None = None,
         fallback_function: Optional[Callable] = None
    ):
        """
        Инициализирующий метод.

        Args:
            failure_count: количество провальных обращений к функции, после чего переходим в состояние Open.
            success_count: количество успешных обращений к функции, после чего переходим в состояние Closed.
            timeout: таймаут, который выжидаем перед новыми попыткамми обратиться к функции.
            Во время таймаута возвращаем заготовленный ответ либо ошибку.
            handle_exceptions: перехватываемые исключения. По умолчанию - Exceptions
            excluded_exceptions: исключения, которые не нужно обрабатывать.
            fallback_function: альтернативное действие, которое можно совершать в случае закрытого состояния или ошибки.
        """
        self._state = BreakerState.CLOSED

        self._failure_counter = 0
        self._max_failure_count = failure_count or self.FAILURE_COUNT

        self._success_counter = 0
        self._min_success_count = success_count or self.SUCCESS_COUNT

        self._timeout = timeout.total_seconds() if timeout else self.TIMEOUT
        self._handle_exceptions = handle_exceptions or self.HANDLE_EXCEPTIONS
        self._excluded_exceptions = excluded_exceptions or self.EXCLUDED_EXCEPTIONS
        self._fallback_function = fallback_function or self.FALLBACK_FUNCTION

        self.name = None

        self._start_time_open_state = None

    @property
    def time_left_in_open_state(self) -> float:
        """Количество времени, которое осталось до выхода CicuitBreker из открытого состояния."""
        if not self._start_time_open_state:
            return 0

        delta = monotonic() - self._start_time_open_state

        if delta > self._timeout:
            return 0

        return self._timeout - delta

    @property
    def is_opened(self) -> bool:
        """Circuit breaker открыт."""
        return self._state == BreakerState.OPEN

    @property
    def is_closed(self) -> bool:
        """Circuit breaker закрыт."""
        return self._state == BreakerState.CLOSED

    @property
    def is_half_opened(self) -> bool:
        """Circuit breaker наполовину открыт."""
        return self._state == BreakerState.HALF_OPEN

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            if self._is_need_handle_error(exc_type):
                self._error_pipeline()
        else:
            self._success_pipeline()

        return False

    def decorator(self, func: callable):
        """Декоратор"""
        self.name = func.__qualname__ if not self.name else self.name

        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.is_opened:
                if self._is_func_exec_after_open_state_finished():
                    self._switch_state(BreakerState.HALF_OPEN)
                elif self._fallback_function:
                    return self._fallback_function(*args, **kwargs)
                else:
                    raise CircuitOpenError(self)

            return self._execute(func, *args, **kwargs)

        return wrapper

    def _is_need_handle_error(self, exc_type) -> bool:
        """
        Метод проверяет, нужно ли обрабатывать исключение.

        Args:
            exc_type: обрабатываемое исключение.
        """
        if self._excluded_exceptions:
            is_need = (not issubclass(exc_type, self._excluded_exceptions)
                       and issubclass(exc_type, self._handle_exceptions))
        else:
            is_need = issubclass(exc_type, self._handle_exceptions)

        return is_need

    def _execute(self, func, *args, **kwargs):
        """
        Запуск функции с помощью контекстного менеджера.

        Помогает при любом исходе отследить завершился ли вызов функции с ошибкой или нет.
        """
        with self:
            return func(*args, **kwargs)

    def _error_pipeline(self):
        """Действия, которые совершаются с CircuitBreaker при ошибочном вызове функции."""
        if self.is_closed:
            self._increase_failure_counter()
            if self._failure_limit_is_reached():
                self._switch_state(BreakerState.OPEN)
        elif self.is_half_opened:
            self._reset_success_counter()
            self._switch_state(BreakerState.OPEN)

    def _success_pipeline(self):
        """Действия, которые совершаются с CircuitBreaker при успешном вызове функции."""
        if self.is_half_opened:
            self._increase_success_counter()
            if self._success_limit_is_reached():
                self._switch_state(BreakerState.CLOSED)

    def _switch_state(self, new_state: BreakerState):
        """
        Метод осуществляет смену состояния CircuitBreaker.

        В зависимости от нового состояния осуществляет изменение зависимых атрибутов CircuitBreaker.
        """
        self._state = new_state

        if self._state == BreakerState.OPEN:
            self._activate_start_time_open_state()

        elif self._state == BreakerState.CLOSED:
            self._reset_failure_counter()

        elif self._state == BreakerState.HALF_OPEN:
            self._reset_success_counter()

    def _is_func_exec_after_open_state_finished(self) -> bool:
        """
        Метод проверяет, запустилась ли функция после выхода из открытого состояния, или нет.

        Returns:
            True - запуск после выхода из открытого состояния, False - запуск во время открытого состояния.
        """
        if not self._start_time_open_state:
            return True

        delta = monotonic() - self._start_time_open_state

        if delta > self._timeout:
            return True

        return False

    def _activate_start_time_open_state(self):
        """Активация таймера открытого состояния"""
        self._start_time_open_state = monotonic()

    def _failure_limit_is_reached(self) -> bool:
        """
        Метод говорит о том, достигнуто ли максимальное количество попыток провального вызова декорируемой функции.

        Returns:
            True - достигнуто, False - не достигнуто.
        """
        return self._failure_counter >= self._max_failure_count

    def _success_limit_is_reached(self):
        """
        Метод говорит о том, достигнуто ли максимальное количество попыток успешного вызова декорируемой функции.

        Returns:
            True - достигнуто, False - не достигнуто.
        """
        return self._success_counter >= self._min_success_count

    def _reset_failure_counter(self):
        """Обнуление количества попыток провального вызова функции."""
        self._failure_counter = 0

    def _reset_success_counter(self):
        """Обнуление количества попыток успешного вызова функции."""
        self._success_counter = 0

    def _increase_failure_counter(self):
        """Увеличение провального количества попыток вызова функции на 1."""
        self._failure_counter += 1

    def _increase_success_counter(self):
        """Увеличение успешного количества попыток вызова функции на 1."""
        self._success_counter += 1


class CircuitOpenError(Exception):
    """Ошибка, которую необходимо выбрасывать в том случае, если CircuitBreaker находится в открытом состоянии."""

    def __init__(self, breaker: CircuitBreaker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._circuit_breaker = breaker

    def __str__(self):
        return 'Circuit находится в открытом состояниии. Запросы к функции будут недоступны еще {left} секунд'.format(
            left=self._circuit_breaker.time_left_in_open_state,
        )


def circuit_breaker(
    failure_count: int | None = None,
    success_count: int | None = None,
    timeout: timedelta | None = None,
    handle_exceptions: tuple | None = None,
    excluded_exceptions: tuple | None = None,
    fallback_function: Optional[Callable] = None
):
    """
    Декоратор, который применяет паттерн Circuit breaker к функции.

    Args:
        failure_count: количество провальных обращений к функции, после чего переходим в состояние Open.
        success_count: количество успешных обращений к функции, после чего переходим в состояние Closed.
        timeout: таймаут, который выжидаем перед новыми попыткамми обратиться к функции.
        Во время таймаута возвращаем заготовленный ответ либо ошибку.
        handle_exceptions: перехватываемые исключения. По умолчанию - Exceptions.
        excluded_exceptions: исключения, которые не нужно обрабатывать.
        Только для указанных иселючений будет производится переход в открытое и полуоткрытое состояние.
        fallback_function: альтернативное действие, которое можно совершать,
        когда CircuitBreaker находится в открытом состоянии.
    """
    def decorator(func):
        return CircuitBreaker(
            failure_count,
            success_count,
            timeout,
            handle_exceptions,
            excluded_exceptions,
            fallback_function,
        ).decorator(func)

    return decorator
