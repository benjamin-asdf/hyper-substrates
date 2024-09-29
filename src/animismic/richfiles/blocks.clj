(comment


  (defn add-blocks! []
    (swap! lib/event-queue
           (fnil conj [])
           (fn [s]
             (lib/append-ents
              s
              (repeatedly
               20
               (fn []
                 (lib/->entity
                  :rect
                  {:color (defs/color-map
                            (rand-nth [:hit-pink :deep-pink
                                       :green-yellow :white
                                       :cyan]))
                   :block? true
                   :mass 1
                   :moment-of-inertia 1000
                   ;; :collides? true
                   :on-collide-map
                   {:die
                    (fn [e other s k]
                      (assoc e :lifetime 1))}
                   ;; :particle? true
                   ;; :kinetic-energy 1
                   :transform (lib/->transform
                               (lib/rand-on-canvas-gauss 0.5)
                               20
                               50
                               1)})))))))




  (add-blocks!)


  (defn update-blocks
    [op]
    (swap! lib/event-queue (fnil conj [])
           (fn [s]
             (lib/update-ents s (fn [e]
                                  (if-not (:block? e) e (op e)))))))
  (update-blocks
   (fn [e]
     (assoc e :color defs/white)))


  (update-blocks
   (fn [e]
     (assoc-in
      e
      [:on-collide-map :foo ]
      (fn [e other s k ]
        (->
         e
         (assoc :color defs/white)
         (assoc :scale 2))))))

  (update-blocks
   (fn [e]
     (assoc
      e
      {:on-collide-map
       {:foo
        (fn [e other s k ]
          (->
           e
           (assoc :color defs/white)
           (assoc :scale 2)))}})))



  (update-blocks
   (fn [e]
     (assoc
      e
      :collides? true
      )))






  (update-blocks
   (fn [e]
     (assoc e :color (:deep-pink defs/color-map))))

  (update-blocks
   (fn [e]
     (lib/live
      e
      [:a
       (lib/every-n-seconds
        1
        (fn [e s k]
          (assoc e :color (defs/color-map
                            (rand-nth [:white :black])))))])))
  (update-blocks
   (fn [e]
     (lib/live
      e
      (let [sin (elib/sine-wave-machine 2 1000)]
        (fn [e s k]
          (let [v (sin)]
            (-> e
                (assoc-in
                 [:transform :scale]
                 (sin)))))))))




  (update-blocks
   (fn [e]
     (lib/live
      e
      (let [sin (elib/sine-wave-machine 10 500)]
        (fn [e s k]
          (let [v (sin)]
            (-> e
                (assoc-in
                 [:transform :scale]
                 (* 10 (sin))))))))))

  (def small? (atom false))















  (swap! lib/event-queue (fnil conj [])

         (fn [s]
           (swap! small? not)
           (lib/update-ents s (fn [e]
                                (update-in e  [:transform :scale] *
                                           (if @small? 0.5 2))))))

  (lib/state-on-update!
   (lib/every-n-seconds
    (fn [] (* 2 (fm.rand/frand))
      ;; (lib/normal-distr 0.2 0.1)
      )
    (fn [s k]
      (swap! small? not)
      (lib/update-ents s (fn [e]
                           (update-in e  [:transform :scale] *
                                      (if @small? 0.5 2))))))))
