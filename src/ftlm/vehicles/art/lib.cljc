(ns ftlm.vehicles.art.lib
  (:require
   [quil.core :as q :include-macros true]
   [fastmath.core :as fm]
   [fastmath.random :as fm.rand]
   [ftlm.vehicles.art.collider :as collide]
   [ftlm.vehicles.art.defs :as defs]
   [ftlm.vehicles.art.util :as u])
  #?(:cljs (:require-macros [ftlm.vehicles.art.util :as
                             u])))

(def ^:dynamic *dt* nil)

(defonce the-state (atom {:event-q (atom [])}))


(defn normal-distr [mean std-deviation]
  (+ mean (* std-deviation (q/random-gaussian))))

(defn v* [[a b] [a' b']]
  [(* a a')
   (* b b')])

(defn v*-1 [[a b] s]
  [(* a s)
   (* b s)])

(defn v+ [[a b] [a' b']]
  [(+ a a')
   (+ b b')])

(defn v- [[a b] [a' b']]
  [(- a a')
   (- b b')])

(defn v-sum [v1 v2]
  (mapv + v1 v2))

(defn signum [x]
  (cond
    (<= x 0) -1
    (> x 0) 1))


(defn controls []
  (q/state :controls))

(defonce eid-counter (atom 0))
(let [->eid #(swap! eid-counter inc)]
  (defn ->entity
    ([kind opts] (merge (->entity kind) opts))
    ([kind] {:id (->eid) :kind kind :spawn-time (q/millis) :entity? true :world :default})))

(defn ->transform [pos width height scale]
  {:pos pos :width width :height height :scale scale :rotation 0})

(def entities (comp vals :eid->entity))
(def entities-by-id #(:eid->entity % {}))

(defn destroy [state eid]
  (update state :eid->entity dissoc eid))

(defn run-or-kill
  [f]
  (fn [e]
    #?(:clj
       (try
         (f e)
         (catch Exception ex
           (println ex)
           (if (map? e)
             (assoc e :kill? true)
             {:kill? true})))
       :cljs
       (try
         (f e)
         (catch js/Error ex
           (println ex)
           (if (map? e)
             (assoc e :kill? true)
             {:kill? true}))))))

(defn update-ents [state f]
  (update state :eid->entity (fn [s] (update-vals s (run-or-kill f)))))

(defn update-ents-parallel
  [state f]
  (update state
          :eid->entity
          (fn [s]
            (into {}
                  (pmap (fn [[k v]] [k ((run-or-kill f) v)])
                        s)))))

(defn validate-entity
  [e]
  (if-not (:entity? e)
    (do (println (str "Not an entity: " (prn-str e))) nil)
    e))

(defn flatten-components
  [ents]
  (let [ents (into ents
                   (comp (mapcat :components)
                         (filter :entity?))
                   ents)
        ents (map (fn [{:as e :keys [components]}]
                    (if (:entity? (peek components))
                      (assoc e
                        :components (map :id components))
                      e))
                  ents)]
    ents))


(defn update--entities
  [state f]
  (update state :eid->entity (fn [s] (into {} (f s)))))

(defn transform [e] (:transform e))
(defn position [e] (-> e transform :pos))
(defn rotation [e] (-> e transform :rotation))
(defn scale [e] (-> e transform :scale))



;; returns a fn that returns the args to q/bezier
(defn ->bezier
  [rel-control-point-1 rel-control-point-2]
  (fn [[x y] [x1 y1 :as end-pos]]
    (let [midpoint (mapv #(/ (+ %1 %2) 2) [x y] end-pos)
          control-point-1 (v+ rel-control-point-1 midpoint)
          control-point-2 (v+ rel-control-point-2 midpoint)]
      [x y (first control-point-1) (second control-point-1)
       (first control-point-2) (second control-point-2) x1
       y1])))

(defn rand-bezier
  [distr]
  (->bezier [(normal-distr 0 distr)
             (normal-distr 0 distr)]
            [(normal-distr 0 distr)
             (normal-distr 0 distr)]))



(def connection->infected :entity-a)
(def connection->non-infected :entity-b)

(defn every-n-seconds [n f]
  (let [n (if (number? n) (constantly n) n)
        till (atom (n))]
    (fn [& args]
      (swap! till - *dt*)
      (when (< @till 0)
        (reset! till (n))
        (apply f args)))))


;; ----------------------------------------

(defonce timers (atom {}))

(defn set-timer
  [seconds]
  (let [seconds (cond (number? seconds) seconds
                      :else (seconds))
        id (random-uuid)]
    (swap! timers assoc id seconds)
    id))

(defn update-timers
  [dt]
  (swap! timers (fn [v]
                  (into {}
                        (comp
                          (map (fn [[k v]] [k (- v dt)]))
                          (filter (fn [[_ v]] (pos? v))))
                        v))))

(defn rang? [timer] (not (@timers timer)))

(defn cooldown
  [n f]
  (let [t (atom nil)]
    (fn [& args]
      (when (rang? @t)
        (reset! t (set-timer n))
        (apply f args)))))

;; ----------------------------------------



;; in ms
(defn age [entity] (- (q/millis) (:spawn-time entity)))

(defn ->hsb-vec
  [color]
  (cond (and (map? color)
             (every? #(contains? color %) [:h :s :b]))
          [(:h color) (:s color) (:b color)]
        (and (map? color)
             (every? #(contains? color %) [:h :s :v :a]))
          [(* (/ (:h color) 360) 255)
           (* 255 (/ (:s color) 100))
           (* 255 (/ (:v color) 100)) (* 255 (:a color))]
        (and (map? color)
             (every? #(contains? color %) [:h :s :v]))
          [(* (/ (:h color) 360) 255)
           (* 255 (/ (:s color) 100))
           (* 255 (/ (:v color) 100))]
        (sequential? color) color
        (nil? color) [0 0 0 0]
        :else [color]))

(defn ->hsb
  [color]
  (let [ret (->hsb-vec color)]
    (apply q/color ret)))

(defn shine
  [{:as entity :keys [shine shinyness]}]

  (if-not shinyness
    entity
    (do
      (q/print-every-n-millisec shinyness)
      (let [shine (mod (+  (or shine 0) (* *dt* shinyness)) 255)]
        (-> entity
            (assoc :shine shine)
            (update :color
                    (fn [_c] (q/color shine 255 255))))))))



(defn sine-wave [frequency time-in-millis]
  (* (Math/sin (* 2 Math/PI (/ time-in-millis 1000) frequency))))

(defn generate-palette [base num-colors]
  (into [] (repeatedly num-colors #(mod (normal-distr base 50) 360))))

(defn *transform [t1 trsf]
  (merge-with * t1 trsf))

(defn update-lifetime-1
  [{:as entity :keys [lifetime]}]
  (if-not lifetime entity (update entity :lifetime - *dt*)))

(defn kill-from-lifetime
  [{:as entity :keys [lifetime]}]
  (if (some-> lifetime (< 0)) (assoc entity :kill? true) entity))

(def update-lifetime (comp kill-from-lifetime update-lifetime-1))

(defn live [e op]
  (update e :on-update-map (fnil conj {}) op))

(defn alive? [e] (and (:entity? e) (not (:kill? e))))

(defn kill-components
  [state]
  (let [kill? (into #{}
                    (comp (remove alive?)
                          (mapcat :components))
                    (entities state))]
    (update-ents
      state
      (fn [{:as e :keys [id]}]
        (if (kill? id) (assoc e :kill? true) e)))))

(defn kill-entities-1
  [state]
  (update state
          :eid->entity
          (fn [m]
            (persistent! (reduce-kv
                           (fn [acc k v]
                             (if (or (:kill? v)
                                     (not (validate-entity
                                            v)))
                               acc
                               (assoc! acc k v)))
                           (transient {})
                           m)))))

(def kill-entities (comp kill-entities-1 kill-components))

(defn update-conn-line
  [{:as entity :keys [connection-line? entity-b entity-a]}
   state]
  (if-not connection-line?
    entity
    (let [e-lut (entities-by-id state)
          dest (e-lut entity-b)
          source (e-lut entity-a)]
      (if-not (and (validate-entity dest)
                   (validate-entity source))
        (assoc entity :kill? true)
        (-> entity
            (assoc-in [:transform :pos] (position source))
            (assoc :end-pos (position dest))
            (assoc :color (:color source)))))))

(def draw-color (comp #(q/fill %) ->hsb))

(defmulti draw-entity :kind)

(defmethod draw-entity :circle
  [{:keys [transform no-stroke? stroke]}]
  (let [[x y] (:pos transform)
        {:keys [width height scale]} transform
        drw (fn []
              (q/ellipse x
                         y
                         (* scale width)
                         (* scale height)))]
    (cond no-stroke? (q/with-stroke nil (drw))
          stroke (q/with-stroke (->hsb stroke) (drw))
          :else (drw))))

(defmethod draw-entity :default [_])


(defmethod draw-entity :line
  [{:keys [transform end-pos color] :as e}]
  (let [[x y] (:pos transform)
        {:keys [_scale]} transform]
    (q/stroke-weight 2)
    (q/with-stroke (->hsb color)
      (q/line [x y] end-pos))))

(defmethod draw-entity :multi-line
  [{:keys [color vertices]}]
  (q/with-stroke (->hsb color)
    (doseq [[p1 p2] (map vector vertices (drop 1 vertices))]
      (q/line p1 p2))))

(defmethod draw-entity :bezier-line
  [{:keys [transform end-pos color bezier-line]}]
  (let [[x y] (:pos transform)
        {:keys [_scale]} transform]
    (q/begin-shape)
    (q/with-stroke (->hsb color)
                   (apply q/bezier
                     (bezier-line [x y] end-pos)))
    (q/end-shape)))

(defmethod draw-entity :triangle
  [{:keys [transform color]}]
  (let [{:keys [pos scale width height rotation]} transform
        [x y] pos
        [w h] [(* scale width) (* scale height)]
        [x1 y1] [0 (- (/ h 2))]
        [x2 y2] [(- (/ w 2)) (+ (/ h 2))]
        [x3 y3] [(+ (/ w 2)) (+ (/ h 2))]]
    (q/stroke-weight 0)
    (q/with-translation
      [x y]
      (q/with-rotation
        [(or rotation 0)]
        (q/with-stroke
          (->hsb [0 0 0])
          (q/with-fill (->hsb color)
            (q/triangle x1 y1 x2 y2 x3 y3)))))))

(defmethod draw-entity :pizza-slice
  [{:keys [transform color]}]
  (let [{:keys [pos scale width height rotation]} transform
        [x y] pos
        [w h] [(* scale width) (* scale height)]
        [x1 y1] [0 (- (/ h 2))]
        [x2 y2] [(- (/ w 2)) (+ (/ h 2))]
        [x3 y3] [(+ (/ w 2)) (+ (/ h 2))]]
    (q/stroke-weight 0)
    (q/with-translation
      [x y]
      (q/with-rotation
        [(or rotation 0)]
        (q/with-stroke (->hsb [0 0 0])
          (q/with-fill
            (->hsb color)
            (q/triangle x1 y1 x2 y2 x3 y3)
            (q/with-translation
              [0 y2]
              (q/ellipse 0 0 w (/ h 2)))))))))

(defmethod draw-entity :rect [{:keys [transform corner-r]}]
  (let [[x y] (:pos transform)
        {:keys [width height scale rotation]} transform]
    (q/with-translation [x y]
      (q/rotate rotation)
      (q/rect 0 0 (* width scale) (* height scale) (or corner-r 0)))))

#?(:cljs
   (defn window-dimensions []
     (let [w (.-innerWidth js/window)
           h (.-innerHeight js/window)]
       [w h])))

(defn rand-on-canvas [] [(rand-int (q/width)) (rand-int (q/height))])

(defn rand-on-canvas-gauss
  [distr]
  [(normal-distr (/ (q/width) 2) (* distr (/ (q/width) 2)))
   (normal-distr (/ (q/height) 2) (* distr (/ (q/height) 2)))])

(def anchor->trans-matrix
  {:top-right [0.8 -1.2]
   :top-left [-0.8 -1.2]
   :top-middle [0 -1.2]
   :middle-middle [0 0]
   :bottom-left [-0.8 1]
   :bottom-right [0.8 1]
   :bottom-middle [0 1]})

(def anchor->rot-influence
  {;; :top-right -1
   ;; :top-left 1
   ;; :top-middle 0
   :bottom-left 1
   :bottom-right -1
   :bottom-middle 0})

;; 2pi * direction
(def anchor->sensor-direction
  {:top-right 0
   :top-left 0
   :top-middle 0
   :bottom-left -1
   :bottom-right -1
   :bottom-middle -1})

(defn +temperature-sensitivity
  [e]
  (-> e
      (assoc :collides? true)
      (assoc-in [:on-collide-map :temperature-sensor]
                (fn [e other state k]
                  (when (and (:temperature-bubble? other)
                             (= (:hot-or-cold e)
                                (:hot-or-cold other)))
                    ;; ------------------------------------------------
                    (update e
                            :activation
                            (fnil + 0)
                            (:temp other 0)))))))


(defn ->sensor
  [{:as opts :keys [modality scale]}]
  (let [scale (cond scale scale
                    (= modality :smell) 0.8
                    :else 1)]
    (cond-> (merge (->entity :circle)
                   {:activation-shine true
                    :color (q/color 40 96 255 255)
                    :particle? true
                    :sensor? true
                    :transform
                    (->transform [0 0] 20 20 scale)}
                   opts)
      (= modality :temperature)
      (+temperature-sensitivity))))

(defn ->motor
  [{:as opts :keys [scale width height]}]
  (let [scale (or scale 1)
        width (or width 20)
        height (or height 35)]
    (merge (->entity :rect)
           {:activation-shine true
            :actuator? true
            :color (q/color 40 96 255 255)
            :motor? true
            :transform
            (->transform [0 0] width height scale)}
           opts)))

(defn ->neuron
  [opts]
  (merge (->entity :neuron) {:activation 0 :hidden? true :neuron? true} opts))

(defn ->baseline-arousal [power]
  (fn [e _]
    (update e :activation + (normal-distr power power))))

(defn ->cap-activation
  ([] (->cap-activation 0))
  ([at]
   (fn [e _]
     (update e :activation
             (fnil #(max at %) 0)))))

;; (update {} :foo (fnil #(max 10 %) 0))

(defn normalize-value-1
  [min max value]
  (let [old-min min
        old-max max
        new-min 0
        new-max 1]
    (+ new-min (* (/ (- value old-min) (- old-max old-min)) (- new-max new-min)))))

(defn relative-position [parent ent]
  (let [{:keys [width height]} (transform parent)
        m
        (or
         (:anchor-position ent)
         (anchor->trans-matrix (:anchor ent))
         [1 1])]
    (v* m [(/ width 2) (/ height 2)])))

;; (relative-position parent ent)

(defn rotate-point [rotation [x y]]
  [(+ (* x (Math/cos rotation)) (* -1 y (Math/sin rotation)))
   (+ (* x (Math/sin rotation)) (* y (Math/cos rotation)))])

(defn translate-point [x y dx dy]
  [(+ x dx) (+ y dy)])

(defn scale-point [[x y] scale]
  [(x * scale) (y * scale)])

(defn friction-1 [velocity]
  (* velocity 0.9))

(defn friction [e]
  (-> e
      (update :velocity (fnil friction-1 0))
      (update :acceleration (fnil friction-1 0))
      (update :angular-velocity (fnil friction-1 0))
      (update :angular-acceleration (fnil friction-1 0))))

(defn track-components
  [state]
  (let [parent-by-id
          (into {}
                (comp
                 ;; (remove :hidden?)
                 (filter :components)
                 (mapcat (fn [ent]
                           (map (juxt identity
                                      (constantly ent))
                                (:components ent)))))
                (entities state))]
    (->
      state
      (update-ents-parallel
        (fn [{:as ent :keys [id]}]
          (if-let [parent (parent-by-id id)]
            (let [relative-position
                  (relative-position parent ent)
                  parent-rotation
                  (or (-> parent
                          :transform
                          :rotation)
                      1)
                  parent-scale
                  (or (-> parent
                          :transform
                          :scale)
                      1)
                  scale
                  (or
                   (-> ent
                       :transform
                       :absolute-scale)
                   (* (or (-> ent
                              :transform
                              :scale)
                          1)
                      parent-scale))]
              (-> ent
                  (assoc-in
                   [:transform :pos]
                   [(+ (first (v* [parent-scale
                                   parent-scale]
                                  (rotate-point
                                   parent-rotation
                                   relative-position)))
                       (first (-> parent
                                  :transform
                                  :pos)))
                    (+ (second (v* [parent-scale
                                    parent-scale]
                                   (rotate-point
                                    parent-rotation
                                    relative-position)))
                       (second (-> parent
                                   :transform
                                   :pos)))])
                  (assoc-in [:transform :rotation]
                            (-> parent
                                :transform
                                :rotation))
                  (assoc-in [:transform :scale] scale)))
            ent))))))

(defn append-ents
  [state ents]
  (let [ents (flatten-components ents)]
    (-> state
        (update
          :eid->entity
          merge
          (into {} (map (juxt :id validate-entity)) ents))
        track-components)))

(defn draw-entities-1
  [entities]
  (doseq [{:as entity :keys [color draw-functions]}
          (sort (u/by (some-fn :z-index (constantly 0))
                      u/ascending
                      :id
                      u/ascending)
                (sequence (comp
                           (remove :hidden?)
                           (filter validate-entity))
                          entities))]
    (q/stroke-weight (or (:stroke-weight entity) 1))
    (draw-color color)
    (let
        [drw (fn []
               (draw-entity entity)
               ;; (doall (map (fn [op] (op entity))
               ;;             (vals draw-functions)))
               )]
        (cond (:stroke entity)
              (q/with-stroke (->hsb (:stroke entity)) (drw))
              :else (drw)))))

(defn draw-entities
  [state]
  (draw-entities-1 (entities state)))

(defn effector->angular-acceleration
  [{:keys [anchor activation rotational-power]}]
  (* (or rotational-power 0) (or activation 0) (anchor->rot-influence anchor)))

(defn update-body
  [entity state]
  (if-not (:body? entity)
    entity
    (do
      (let [effectors (sequence (comp
                                   (map (entities-by-id state))
                                   (filter :motor?))
                                  (:components entity))]
          (-> entity
              (update :acceleration
                      (fnil + 0)
                      (reduce +
                        (map #(:activation % 0) effectors)))
              (assoc :angular-acceleration
                       (transduce
                         (map
                           effector->angular-acceleration)
                         +
                         effectors)))))))

(defn update-rotation
  [entity]
  (let [angular-velocity
          (+ (:angular-velocity entity 0)
             (* *dt* (:angular-acceleration entity 0)))]
    (-> entity
        (update-in [:transform :rotation]
                   (fnil #(+ % angular-velocity) 0))
        (assoc :angular-velocity angular-velocity))))

(defn move-dragged
  [entity]
  (if (:dragged? entity)
    (assoc-in entity [:transform :pos] [(q/mouse-x) (q/mouse-y)])
    entity))

(defn update-position
  [{:as entity :keys [velocity acceleration]}]
  (if-not (-> entity :transform :pos)
    entity
    (let [velocity (or velocity 0)
          acceleration (or acceleration 0)
          velocity (+ velocity (* *dt* acceleration))
          rotation (-> entity
                       :transform
                       (:rotation 0))
          x (* *dt* velocity (Math/sin rotation))
          y (* *dt* velocity (- (Math/cos rotation)))]
      (-> entity
          (assoc :velocity velocity)
          (update-in [:transform :pos]
                     (fn [position]
                       (vector (+ (first position) x)
                               (+ (second position)
                                  y))))))))


(defn activation-shine
  [{:as entity
    :keys [activation shine activation-shine-colors
           activation-shine activation-shine-speed]}]
  ;; The lerp color part is relatively expensive,
  ;; if I have 1000+ entities
  ;;
  (if (:hidden? entity)
    entity
    (if (and activation activation-shine)
      (let [shine (or shine 0)
            shine (+ shine
                     (* *dt*
                        activation
                        (or activation-shine-speed 1)))]
        (assoc entity
          :shine shine
          :color
            ;; defs/white
            (q/lerp-color
              (->hsb (or (:low activation-shine-colors)
                         (q/color 40 96 255 255)))
              (->hsb (or (:high activation-shine-colors)
                         (q/color 100 255 255)))
              (normalize-value-1 0 1 (Math/sin shine)))))
      entity)))

(defmulti update-sensor (fn [sensor _env] (:modality sensor)))

(defn distance
  [[x1 y1] [x2 y2]]
  (Math/sqrt (+ (Math/pow (- x2 x1) 2) (Math/pow (- y2 y1) 2))))

(defn normalize-value
  [min max value]
  (let [range (- max min)]
    (+ min (mod (/ (- value min) range) 1))))

(defn position-on-circle
  [center radius angle]
  (let [rad (q/radians angle)]
    [(+ (first center) (* radius (Math/cos rad)))
     (+ (second center) (* radius (Math/sin rad)))]))

(defn point-inside-circle?
  [[x y] [ox oy] d]
  (let [radius (/ d 2)]
    (<= (+ (Math/pow (- x ox) 2)
           (Math/pow (- y oy) 2))
        (Math/pow radius 2))))

(defn point-inside-rect?
  [[x y] [x1 y1] [x2 y2]]
  (and (<= x1 x x2) (<= y1 y y2)))

(defmulti point-inside? (fn [e point] (:kind e)))

(defmethod point-inside? :circle
  [circle point]
  (let [[x y] point
        [ox oy] (position circle)
        d (-> circle :transform :width)]
    (point-inside-circle? [x y] [ox oy] d)))

(defmethod point-inside? :rect
  [rect point]
  (let [[x y] point
        [ox oy] (position rect)
        {:keys [width height scale rotation]} (:transform rect)
        [x1 y1] [(- ox (/ width 2)) (- oy (/ height 2))]
        [x2 y2] [(+ ox (/ width 2)) (+ oy (/ height 2))]]
    (point-inside-rect?
     [x y]
     [x1 y1]
     [x2 y2])))

(defn update-sensors
  [entity env]
  (if (:sensor? entity) (update-sensor entity env) entity))

(defn activation-decay [{:keys [activation] :as entity}]
  (if activation
    (let [sign (signum activation)
          activation (* sign (- (abs activation) 0.2) 0.8)]
      (assoc entity :activation activation))
    entity))

(comment
  (let [activation 14
        sign (signum activation)
        activation (* sign (- (abs activation) 0.2) 0.8)]
    activation)
  (let [activation 11.040000000000001
        sign (signum activation)
        activation (* sign (- (abs activation) 0.2) 0.8)]
    activation))

(defn ->transdution-model
  ([a b] (->transdution-model a b identity))
  ([a b f] {:source a :destination b :f (or f identity)}))

(def excite #(* 1 %))
(def inhibit #(* -1 %))
(defn ->weighted [weight] #(* weight %))

(def connection->source (comp :source :transduction-model))
(def connection->destination (comp :destination :transduction-model))

(defn transduce-signal
  [destination source {:keys [f gain]}]
  (let [gain (cond (number? gain) #(* gain %)
                   (fn? gain) gain
                   :else identity)]
    (update destination
            :activation
            (fnil + 0)
            (gain (f (:activation source 0))))))

(defn transduce-signals
  [state]
  (let [models (keep :transduction-model (entities state))
        e-lut (:eid->entity state)
        new-lut (reduce (fn [lut model]
                          (update lut
                                  (:destination model)
                                  transduce-signal
                                  (e-lut (:source model))
                                  model))
                        e-lut
                        models)]
    (assoc state :eid->entity new-lut)))

(defn mid-point [] [(/ (q/width) 2)
                    (/ (q/height) 2)])

(defn angle-between [[x1 y1] [x2 y2]] (- (Math/atan2 (- y1 y2) (- x1 x2)) q/HALF-PI))

(defn normalize-angle [angle]
  (mod (+ (mod angle q/TWO-PI) q/TWO-PI) q/TWO-PI))

(defn orient-towards
  [entity target]
  ;; (def entity entity)
  ;; (def target target)
  (let [desired-angle (angle-between (position entity) target)]
    (assoc-in entity [:transform :rotation] desired-angle)))

(defn ->grow
  [speed]
  (fn [e _]
    (cond-> (update-in e [:transform :scale] + (* *dt* speed))
      (-> e :transform :absolute-scale)
      (update-in [:transform :absolute-scale] + (* *dt* speed)))))

(defn ->clamp-scale
  [max]
  (fn [e _]
    (update-in e [:transform :scale] #(min max %))))

;; sensors have 180 degree of seeing.
;; 1 directly in front 0 to the side, 0 when behind
(defn calculate-adjustment [angle looking-direction]
  (max 0 (Math/cos (+ (* looking-direction q/TWO-PI) angle))))

(defn ray-intensity
  [sensor-pos sensor-rotation sensor-looking-direction env]
  (transduce
   (map (fn [light]
          (let [distance (distance sensor-pos (position light))
                angle-to-source (angle-between (position light) sensor-pos)
                relative-angle (- sensor-rotation angle-to-source)
                angle (- relative-angle q/PI)
                raw-intensity (/ (:intensity light)
                                 (/ (* distance distance) 5000))
                adjustment (calculate-adjustment angle
                                                 sensor-looking-direction)]
            (* raw-intensity adjustment))))
   +
   (:ray-sources env)))

(defmethod update-sensor :rays
  [sensor env]
  (let [ray-intensity (ray-intensity
                       (position sensor)
                       (rotation sensor)
                       (-> sensor
                           :anchor
                           anchor->sensor-direction)
                       env)]
    (assoc sensor :activation (min ray-intensity 14))))

(defn ->odor-source
  [{:keys [intensity decay-rate pos] :as opts}]
  (merge
   (->entity :odor)
   {:odor-source? true
    :intensity intensity
    :decay-rate decay-rate
    :hidden? true?
    :transform (->transform pos 0 0 0)}
   opts))

(defmethod update-sensor :smell
  [sensor env]
  (let
      ;; odor sensor activiy is a function of the distance
      ;; only. So smell doesn't have a direction. Inverse power law
      [new-activation
       (transduce
        (comp
         (filter (comp (:fragrance sensor) :fragrances))
         (map (fn [odor-source]
                (let [distance (distance (position sensor)
                                         (position odor-source))]
                  (/
                   (* 200 (:intensity odor-source))
                   (* distance distance (:decay-rate odor-source)))))))
        +
        (-> env :odor-sources))]
      (assoc sensor :activation (min new-activation 14))))

#_(defmethod update-sensor :temperature
  [sensor env]
  (let [new-activation
        (transduce
         (comp (filter (comp #{(:hot-or-cold sensor)}
                             :hot-or-cold))
               (filter (fn [{:as bubble :keys [d]}]
                         (point-inside-circle?
                          (position sensor)
                          (position bubble)
                          d)))
               (map :temp))
         +
         (-> env
             :temperature-bubbles))]
    (assoc sensor :activation (min new-activation 14))))

(defmethod update-sensor :temperature [sensor env] sensor)

(defn ->circular-shine-1
  [e]
  (let [pos (position e)]
    (assoc (->entity :circle)
           :transform (assoc (->transform pos 20 20 1)
                             :absolute-scale 1)
           :lifetime 1
           :color (defs/color-map
                    (rand-nth [:hit-pink :red :heliotrope
                               :green-yellow :horizon :magenta
                               :purple :sweet-pink :cyan]))
           :z-index -4)))

(defn ->circular-shine
  ([freq speed]
   (->circular-shine freq speed ->circular-shine-1))
  ([freq speed make-shine]
   (->circular-shine freq speed 3 make-shine))
  ([freq speed lifetime make-shine]
   (let [s (atom {:next freq})
         freq (/ freq 3)]
     (fn [entity state]
       (swap! s update :next - *dt*)
       (when (<= (:next @s) 0)
         (swap! s assoc :next (normal-distr freq freq))
         (let [c (->hsb (:color entity))
               se (-> (assoc (make-shine
                               entity
                               #_(q/color (q/hue c)
                                          (q/saturation c)
                                          (q/brightness c)
                                          100))
                               :on-update
                             [(->grow speed)])
                      (update :lifetime
                              (fn [l]
                                (or l
                                    (normal-distr
                                      lifetime
                                      (Math/sqrt
                                        lifetime))))))]
           {:updated-state (-> state
                               (update-in [:eid->entity
                                           (:id entity)
                                           :components]
                                          (fnil conj [])
                                          (:id se))
                               (append-ents [se]))}))))))


(defn ->explosion
  [{:keys [n size pos color spread]}]
  (into
   []
   (map (fn [_]
          (let [spawn-pos
                [(normal-distr (first pos) spread)
                 (normal-distr (second pos) spread)]]
            (-> (merge
                 (->entity :circle)
                 {:acceleration (normal-distr 1000 200)
                  :color color
                  :lifetime (normal-distr 1 0.5)
                  :transform
                  (assoc
                   (->transform spawn-pos size size 1)
                   :rotation (angle-between spawn-pos
                                            pos))})))))
   (range n)))

(defn +explosion
  [state e]
  (append-ents state
               (->explosion {:color (:color e)
                             :n 20
                             :pos (position e)
                             :size 10
                             :spread 10})))


(defn ->wobble-anim
  [duration magnitute]
  (let [s (atom {:time-since 0})
        inner
          (fn [e]
            (if (= :done @s)
              e
              (do
                (when-not (:initial-scale @s)
                  (swap! s assoc
                    :initial-scale
                    (-> e
                        :transform
                        :scale)))
                (swap! s update :time-since + *dt*)
                (let [progress (/ (:time-since @s) duration)
                      initial-scale (:initial-scale @s)]
                  (if (< 1.0 progress)
                    (do (reset! s :done)
                        (assoc-in e
                          [:transform :scale]
                          initial-scale))
                    (assoc-in e
                      [:transform :scale]
                      (q/lerp
                        (:initial-scale @s)
                        (* (:initial-scale @s) magnitute)
                        (q/sin (float (* q/PI
                                         progress))))))))))]
    (fn ([e _state] (inner e)) ([e _state _k] (inner e)))))

(defn wobble [e duration magnitute]
  (live e
        [:wobble
         (->wobble-anim duration magnitute)]))

(defn burst
  [e _other state _]
  {:updated-state
   (-> (+explosion state e)
       (update-in [:eid->entity (:id e)] wobble 1 3))})

(defn ->ray-source
  [{:as opts :keys [pos intensity scale shinyness]}]
  [(merge (->entity :circle)
          {:collides? true
           :color {:h 67 :s 7 :v 95}
           :draggable? true
           :on-collide-map
             {:burst
                (cooldown
                  2
                  (fn [e _other state _]
                    {:updated-state
                       (cond-> (+explosion state e)
                         :wobble (update-in [:eid->entity
                                             (:id e)]
                                            wobble
                                            1
                                            3)
                         (-> state
                             :controls
                             :ray-sources-die?)
                           (assoc-in [:eid->entity (:id e)
                                      :lifetime]
                             0.8))}))}
           :particle? true
           :ray-source? true
           :shinyness
             (if-not (nil? shinyness) shinyness intensity)
           :transform (assoc (->transform pos 40 40 1)
                        :scale (or scale 1))}
          opts)])

(defn ->body
  [{:as opts :keys [pos scale rot]}]
  (merge (->entity :rect)
         {:body? true
          ;; 'physical'
          :collides? true
          :transform (assoc (->transform pos 50 80 scale)
                       :rotation rot)}
         opts))


;; or clickable
(defn find-closest-draggable
  [state]
  (let [mouse-position [(q/mouse-x) (q/mouse-y)]]
    (->> state
         entities
         (filter (some-fn :draggable? :clickable?))
         (sort-by (comp (fn [b] (distance mouse-position b)) position))
         (first))))

(defn track-conn-lines
  [state]
  (update-ents state #(update-conn-line % state)))

(defn dart-to-middle
  [{:as entity :keys [darts?]}]
  (if (and darts?
           (< (normal-distr 1000 200)
              (- (q/millis) (get entity :last-darted -500))))
    (-> entity
        (orient-towards (mid-point))
        (assoc :acceleration 100)
        (assoc :last-darted (q/millis)))
    entity))

(defn inside-screen?
  [[x y]]
  (and (< 0 x (q/width))
       (< 0 y (q/height))))

(defn kinetic-energy-motion
  [entity kinetic-energy]
  (-> entity
      (update :acceleration
              (fnil + 0)
              (* 30 (q/random-gaussian) kinetic-energy))
      (update :angular-acceleration
              (fnil + 0)
              (* 0.3 (q/random-gaussian) kinetic-energy))))

(defn calculate-center-point [entities]
  (let [positions (map position entities)
        count (count positions)
        [x y] (reduce (fn [[sum-x sum-y] [x y]] [(+ sum-x x) (+ sum-y y)]) [0 0] positions)]
    [(/ x count)
     (/ y count)]))

(defn brownian-motion
  [e]
  (if-not (:particle? e)
    e
    (kinetic-energy-motion e
                           (or (:kinetic-energy e)
                               (:brownian-factor (controls))
                               0))))

(defn ray-source-collision-burst
  [state]
  (let [sources (sequence (comp (filter :ray-source?)
                                (filter (comp
                                         #(< 1000 %)
                                         #(- (q/millis) %)
                                         (fnil identity 0)
                                         :last-exploded)))
                          (entities state))
        bodies (filter :body? (entities state))
        explode-them (into #{}
                           (comp (remove
                                  (fn [[s b]]
                                    (< (* (scale s) 100)
                                       (distance
                                        (position s)
                                        (position b)))))
                                 (map first)
                                 (map :id))
                           (for [source sources
                                 body bodies]
                             [source body]))]
    (-> state
        (update :eid->entity
                (fn [lut]
                  (reduce (fn [m id]
                            (cond-> m
                              :always (assoc-in
                                       [id :last-exploded]
                                       (q/millis))
                              :always (update-in
                                       [id :on-update]
                                       conj
                                       (->wobble-anim 1 3))
                              (-> state
                                  :controls
                                  :ray-sources-die?)
                              (assoc-in [id :lifetime]
                                        0.8)))
                          lut
                          explode-them)))
        (append-ents
         (into []
               (comp (map (entities-by-id state))
                     (map (fn [e]
                            (->explosion {:color (:color e)
                                          :n 20
                                          :pos (position e)
                                          :size 10
                                          :spread 10})))
                     cat)
               explode-them)))))

(defn update-collision
  [state entity-source entity-target]
  (reduce (fn [s [k f]]
            (let [{:as e :keys [updated-state]}
                    ;;
                    ;; Collide is a function of
                    ;; [ entity other state key ]
                    ;;
                    (f ((entities-by-id s) entity-target)
                       ((entities-by-id s) entity-source)
                       s
                       k)]
              (cond updated-state updated-state
                    e (assoc-in s
                        [:eid->entity entity-target]
                        e)
                    :else s)))
    state
    (:on-collide-map ((entities-by-id state)
                       entity-target))))

(defmulti radius :kind)

(defmethod radius :default
  [e]
  (* (-> e
         :transform
         (:scale 0))
     (min
      (-> e
          :transform
          (:width 0))
      (-> e
          :transform
          (:height 0)))))

(defmethod radius :circle
  [e]
  (def e e)
  (* (-> e
         :transform
         (:scale 0))
     (/ (max (-> e
                 :transform
                 (:width 0))
             (-> e
                 :transform
                 (:height 0)))
        2)))

(defn update-collisions
  [state]
  (def state state)
  (let [ents (vec (filter :collides? (entities state)))]
    ;; (let [ents (vec (filter :collides? (entities state)))]
    ;;   (let [ ;; positions
    ;;         positions (map position ents)
    ;;         radii (map radius ents)]
    ;;     (collide/collisions positions radii)))

    (reduce (fn [s [i j]]
              (-> s
                  (update-collision (:id (ents i))
                                    (:id (ents j)))
                  (update-collision (:id (ents j))
                                    (:id (ents i)))))
            state
            (let [ ;; positions
                  positions (map position ents)
                  radii (map radius ents)]
              (collide/collisions positions radii)))))

(defn update-update-functions-1
  [state]
  (transduce (filter :on-update)
             (completing (fn [s {:keys [on-update id]}]
                           (reduce (fn [s f]
                                     (let [{:as e :keys [updated-state]}
                                           (f ((entities-by-id s) id) s)]
                                       (cond updated-state updated-state
                                             e (assoc-in s [:eid->entity id] e)
                                             :else s)))
                                   s
                                   on-update)))
             state
             (entities state)))

(defonce input (atom nil))
(defonce output (atom nil))

(defn update-update-functions-map-1
  [k]
  (fn [state]
    (transduce
     (filter k)
     (completing
      (fn [s {:as e :keys [id]}]
        (let [update-map (k e)]
          (reduce
           (fn [s [k f]]
             (let [{:as e :keys [updated-state]}
                   (f ((entities-by-id s) id) s k)]
               (cond updated-state updated-state
                     e (assoc-in s [:eid->entity id] e)
                     :else s)))
           s
           update-map))))
     state
     (entities state))))

(def update-update-functions-map (update-update-functions-map-1 :on-update-map))
(def update-late-update-map (update-update-functions-map-1 :on-late-update-map))

(defn ->call-callbacks
  [k]
  (fn [state e]
    ;; (def state state)
    ;; (def e e)
    (let [s state
          {:as e :keys [id]} e
          cb-map (k e)]
      (reduce (fn [s [k f]]
                (let [{:as e :keys [updated-state]}
                        (f ((entities-by-id s) id) s k)]
                  (cond updated-state updated-state
                        e (assoc-in s [:eid->entity id] e)
                        :else s)))
        s
        cb-map))))

(def call-double-clicks (->call-callbacks :on-double-click-map))

(defn update-update-functions
  [state]
  (-> state
      update-update-functions-1
      update-update-functions-map))

(defn update-state-update-functions-1
  [{:keys [on-update] :as state}]
  (reduce (fn [s f] (or (f s) s)) state on-update))

(defn update-state-update-functions-map
  [{:keys [on-update-map] :as state}]
  (reduce (fn [s [k f]] (or (f s k) s)) state on-update-map))

(defn update-state-update-functions
  [state]
  (->
   state
   update-state-update-functions-1
   update-state-update-functions-map))

(defn for-n-seconds
  [n f cb]
  (let [n (if (number? n) (constantly n) n)
        till (atom (n))
        done? (atom false)]
    (fn [& args]
      (swap! till - *dt*)
      (when (not @done?)
        (if (< 0 @till)
          (apply f args)
          (do (reset! done? true) (apply cb args)))))))

(def event-queue (atom []))

(defmulti event!
  (fn [e _]
    (def foo e)
    (if (fn? e) :fn (or (:kind e) e))))
(defmethod event! :default [f s] (f s))
(defmethod event! :fn [f s] (f s))

;; (event! (fn [e] e) {})

(defn apply-events
  ([state eventq]
   (reduce (fn [s e] (event! e s))
           state
           (let [r @eventq]
             (reset! eventq [])
             r)))
  ([state] (apply-events state event-queue)))


(defn dart-distants-to-middle
  [{:as entity :keys [darts?]}]
  (if (and
       darts?
       (not (inside-screen? (position entity))))
    (-> entity dart-to-middle)
    entity))

(defn env
  [state]
  (let [ents (entities state)]
    {:odor-sources (filter :odor-source? ents)
     :ray-sources (filter :ray-source? ents)
     :temperature-bubbles (filter :temperature-bubble?
                            ents)}))

(defn ->sub-circle
  [angle radius opts]
  (let [center-pos (:pos opts)
        sub-pos (position-on-circle center-pos radius angle)]
    (->
     (->entity :circle)
     (merge
      {:radius radius :angle angle}
      opts)
     (assoc-in [:transform :pos] sub-pos))))

(defn ->clock-circles
  [center radius count opts]
  (let [angle-step (/ 360 count)]
    (map-indexed
     (fn [idx _]
       (->sub-circle (* idx angle-step) radius (assoc opts :pos center)))
     (range count))))

;; ---------------------------------------------------
;; ---------------------------------------------------
;; this makes triangles look like little gears, amazing
;; :rotate-just-a-little
;;      (every-n-seconds
;;       0.5
;;       (fn [e _ _]
;;         (update e :angular-acceleration + (normal-distr speed (/ speed 2)))))
;; ----------------------------------------------------

(defn ->fade
  []
  (fn [e _ _]
    (update e
            :color
            (fn [c]
              (let [c (->hsb c)
                    a (q/alpha c)]
                (q/color (q/hue c)
                         (q/saturation c)
                         (q/brightness c)
                         (- a (* 100 *dt*))))))))

(defn ->fade-pulse
  [duration]
  (let [s (atom {:time-since 0})]
    (fn [e _ _]
      (swap! s update :time-since + *dt*)
      (let [progress (/ (:time-since @s) duration)]
        ;; (when (< 1.0 progress)
        ;;   (swap! s assoc :time-since 0))
        (update e
                :color
                (fn [c]
                  ;; (println (q/sin (* q/PI progress)))
                  (let [c (->hsb c)
                        new-a (q/lerp 0 255 (q/sin (* q/PI progress)))]
                    (q/color
                     (q/hue c)
                     (q/saturation c)
                     (q/brightness c)
                     new-a))))))))

;; (defn ->fade-pulse-2
;;   [duration]
;;   (let [s (atom {:time-since 0})]
;;     (fn [e _ _]
;;       (swap! s update :time-since + *dt*)
;;       (let [progress (/ (:time-since @s) duration)]
;;         (update
;;           e
;;           :color
;;           (fn [c]
;;             (let [c (->hsb c)
;;                   new-a (q/lerp
;;                           0
;;                           255
;;                           (+ 1 (q/sin (* q/PI progress))))]
;;               (q/color (q/hue c)
;;                        (q/saturation c)
;;                        (q/brightness c)
;;                        new-a))))))))

;; (defn ->wave-function
;;   [duration]
;;   (let [s (atom {:time-since 0})]
;;     (fn []
;;       (swap! s update :time-since + *dt*)
;;       (let [progress (/ (:time-since @s) duration)]
;;         (+ 1 (q/sin (* q/PI progress)))))))

(defn ->fade-pulse-2
  ([duration] (->fade-pulse-2 duration :color))
  ([duration k]
   (let [s (atom {:time-since 0})]
     (fn [e _ _]
       (swap! s update :time-since + *dt*)
       (let [progress (/ (:time-since @s) duration)]
         (update
           e
           k
           (fn [c]
             (let [c (->hsb c)
                   new-a (q/lerp
                          0
                          255
                          (+ 1 (q/sin (* q/PI progress))))]
               (q/color (q/hue c)
                        (q/saturation c)
                        (q/brightness c)
                        new-a)))))))))



(defn with-alpha
  [color a]
  ;; is tech debt, ->hsb and vector
  (let [c (->hsb color)]
    (q/color (q/hue c) (q/saturation c) (q/brightness c)
             (* 255 a)))
  ;; (let [c color]
  ;;   (q/color (q/hue c) (q/saturation c) (q/brightness c)
  ;;            (* 255 a)))
  )

#_(defn grid []
  (let [w (q/width)
        h (q/height)
        x-step 50
        y-step 50]
    (doseq [x (range 0 w x-step)
            y (range 0 h y-step)]
      (q/line x 0 x h)
      (q/line 0 y w y))))

(defn on-double-click
  [state id]
  (call-double-clicks state ((entities-by-id state) id)))

(defn double-clicked? [{id-1 :id time-old :time} {id-2 :id time-new :time}]
  (and
   (= id-2 id-1)
   (< (- time-new time-old) 300)))

(defn inside-canvas?
  [x y]
  (and
   (<= 0 x (q/width))
   (<= 0 y (q/height))))

(defn mouse-pressed
  [state event]
  (if-not (inside-canvas? (q/mouse-x) (q/mouse-y))
    state
    (let [draggable-or-clickable (find-closest-draggable
                                   state)]
      (if draggable-or-clickable
        (let [new-selection {:id (:id
                                   draggable-or-clickable)
                             :time (q/millis)}
              old-selection (:selection state)
              state (-> state
                        (assoc :pressed true)
                        (assoc-in [:eid->entity
                                   (:id
                                     draggable-or-clickable)
                                   :dragged?]
                                  (:draggable?
                                    draggable-or-clickable))
                        (assoc :selection new-selection))
              state ((->call-callbacks :on-click-map)
                      state
                      draggable-or-clickable)
              state (if-not (:draggable?
                              draggable-or-clickable)
                      state
                      ((->call-callbacks :on-drag-start-map)
                        state
                        draggable-or-clickable))]
          (cond-> state
            (double-clicked? old-selection new-selection)
              (on-double-click (:id
                                 draggable-or-clickable))))
        state))))

(defn mouse-released
  [{:as state :keys [selection]} _]
  (def state state)
  (def selection selection)
  (if-not selection
    state
    (let [{:as selection :keys [dragged?]}
            ((entities-by-id state) (:id selection))]
      (if-not selection
        state
        (cond-> state
          dragged?
          ((->call-callbacks :on-drag-end-map) selection)
          :regardless (update-in
                        [:eid->entity (:id selection)]
                        (fn [e]
                          (assoc e :dragged? false))))))))

(defn rotate-entity
  [state id rotation]
  (update-in state [:eid->entity id :transform :rotation] + rotation))

(defn mouse-wheel
  [state rotation]
  (if-let [ent ((entities-by-id state)
                 (-> state
                     :selection
                     :id))]
    (do (rotate-entity state
                       (:id ent)
                       #?(:cljs (/ rotation 50 2.5)
                          :clj (* rotation 100)))
        ;; (update-in state
        ;;            [:eid->entity (:id ent)
        ;;            :angular-acceleration]
        ;;            +
        ;;            (* rotation 100))
    )
    state))

(def actuator? :actuator?)

(defn ->watch-ent
  [state-atom {:keys [id]} f]
  (fn [e s _]
    (f e ((entities-by-id @state-atom) id) s)))

(defn ->derived-entity
  [state-atom world {:keys [id kind]} f]
  (merge (->entity kind)
         {:on-update-map
            {:watch (->watch-ent state-atom {:id id} f)}
          :world world}))

(defn ->suicide-packt
  [others]
  (fn [e s _]
    (if (some (complement alive?)
              (map (entities-by-id s) others))
      (assoc e :kill? true)
      e)))

(defn suicide-packt
  [e others]
  (live e
        [[:suicide-packt (random-uuid)]
         (->suicide-packt (into #{}
                                (map (some-fn :id identity)
                                     others)))]))

(defn apply-update-events
  [state]
  (let [updates (:updates state)]
    ;; update f: a function of 2 args, the state and
    ;; the event
    (->
     (reduce (fn [s {:as evnt :keys [f]}] (f s evnt)) state updates)
     (dissoc :updates))))

;; op: 3 args entity, state, event-data
(defn entity-update-event
  [id op event-data]
  (merge event-data
         {:entity-update? true
          :f (fn [s _]
               (if-let [e ((entities-by-id s) id)]
                 (assoc-in s
                   [:eid->entity id]
                   (op e s event-data))
                 s))
          :id id}))

(defn put [e pos]
  (assoc-in e [:transform :pos] pos))

(defn put-rotation [e rotation]
  (assoc-in e [:transform :rotation] rotation))

(defn clone-entity
  [e]
  (merge (->entity (:kind e) (dissoc e :id))
         {:clone-source (:id e) :clone? true}))


;; (defn cooldown
;;   [n-seconds k op]
;;   (let [left-in-window (atom n-seconds)
;;         event-count-left (atom k)]
;;     (fn [& args]
;;       (print left-in-window *dt*)
;;       (swap! left-in-window - *dt*)
;;       (println @left-in-window)
;;       (if (< 0 @left-in-window)
;;         (when (< 0 @event-count-left)
;;           (swap! event-count-left dec)
;;           (apply op args))
;;         (do
;;           (println "reset")
;;           (reset! left-in-window n-seconds)
;;           (reset! event-count-left k)
;;           nil)))))

(defmulti setup-version (comp keyword :v :controls))

(defn from-left [amount]
  (- (q/width) amount))

(defn from-bottom [amount]
  (- (q/height) amount))

;; ------------------------------------------------

(defn ->connection-line-1
  [entity-a entity-b]
  {:children [(:id entity-b) (:id entity-b)]
   :color (:color entity-a)
   :connection-line? true
   :end-pos (position entity-b)
   :entity-a (:id entity-a)
   :entity-b (:id entity-b)
   :transform (->transform (position entity-a) 1 1 1)})

(defn ->connection-line
  [entity-a entity-b]
  (->entity :line (->connection-line-1 entity-a entity-b)))

(defn ->connection-bezier-line
  [entity-a entity-b]
  (merge (->entity :bezier-line
                   {:bezier-line (rand-bezier 5)})
         (->connection-line-1 entity-a entity-b)))

(defn ->connection
  [{:as opts
    :keys [source destination hidden? bezier-line f]}]
  (-> (merge opts
             (cond hidden? (->entity :hidden-connection)
                   bezier-line (->connection-bezier-line
                                 source
                                 destination)
                   :else (->connection-line source
                                            destination))
             {:connection? true
              :hidden? hidden?
              :transduction-model (->transdution-model
                                    (:id source)
                                    (:id destination)
                                    f)})
      (suicide-packt [source destination])))

(defn kill-some
  [chance]
  (fn [s]
    (update-ents s
                 (fn [e]
                   (if-not (zero? (fm.rand/flip chance))
                     (assoc e :kill? true)
                     e)))))
