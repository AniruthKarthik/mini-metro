import type { StateSnapshot, ActionEnvelope } from '../types';

export type StateListener = (snapshot: StateSnapshot) => void;
export type ErrorListener = (errorMsg: string) => void;
export type ConnectionListener = (connected: boolean) => void;

export class GameWSClient {
  private ws: WebSocket | null = null;
  private url: string;
  private stateListeners: Set<StateListener> = new Set();
  private errorListeners: Set<ErrorListener> = new Set();
  private connectionListeners: Set<ConnectionListener> = new Set();
  private latestSnapshot: StateSnapshot | null = null;
  private isConnected: boolean = false;
  private reconnectTimer: number | null = null;

  constructor(url: string = `ws://${window.location.hostname}:6969/ws`) {
    this.url = url;
  }

  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('🚇 Connected to Mini Metro WebSocket server');
        this.isConnected = true;
        this.notifyConnection(true);
        if (this.reconnectTimer !== null) {
          window.clearTimeout(this.reconnectTimer);
          this.reconnectTimer = null;
        }
      };

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'action_error') {
            console.warn('Action error from server:', data.error);
            this.notifyError(data.error);
            return;
          }

          if (typeof data.tick === 'number') {
            this.latestSnapshot = data as StateSnapshot;
            this.notifyState(this.latestSnapshot);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      this.ws.onerror = (err) => {
        console.error('WebSocket connection error:', err);
      };

      this.ws.onclose = () => {
        console.log('WebSocket connection closed. Attempting reconnect in 2s...');
        this.isConnected = false;
        this.notifyConnection(false);
        this.ws = null;
        this.scheduleReconnect();
      };
    } catch (err) {
      console.error('Failed to instantiate WebSocket:', err);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer === null) {
      this.reconnectTimer = window.setTimeout(() => {
        this.reconnectTimer = null;
        this.connect();
      }, 2000);
    }
  }

  public sendAction(action: ActionEnvelope): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('Cannot send action: WebSocket not connected');
      return false;
    }
    this.ws.send(JSON.stringify(action));
    return true;
  }

  public onState(listener: StateListener): () => void {
    this.stateListeners.add(listener);
    if (this.latestSnapshot) {
      listener(this.latestSnapshot);
    }
    return () => this.stateListeners.delete(listener);
  }

  public onError(listener: ErrorListener): () => void {
    this.errorListeners.add(listener);
    return () => this.errorListeners.delete(listener);
  }

  public onConnection(listener: ConnectionListener): () => void {
    this.connectionListeners.add(listener);
    listener(this.isConnected);
    return () => this.connectionListeners.delete(listener);
  }

  private notifyState(snapshot: StateSnapshot): void {
    for (const listener of this.stateListeners) {
      listener(snapshot);
    }
  }

  private notifyError(msg: string): void {
    for (const listener of this.errorListeners) {
      listener(msg);
    }
  }

  private notifyConnection(connected: boolean): void {
    for (const listener of this.connectionListeners) {
      listener(connected);
    }
  }

  public getSnapshot(): StateSnapshot | null {
    return this.latestSnapshot;
  }
}
